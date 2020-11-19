//===- ExtractScopStmt.cc - Extract scop stmt to func -----------------C++-===//
//
// This file implements the transformation that extracts scop statements into
// MLIR functions.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/ExtractScopStmt.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

using CalleeName = SmallString<16>;

/// Discover the operations that have memory write effects.
/// TODO: use MemoryEffects to properly detect ops that has memory write side
/// effects.
static void discoverMemWriteOps(mlir::FuncOp f,
                                SmallVectorImpl<Operation *> &ops) {
  f.getOperation()->walk([&](Operation *op) {
    if (isa<mlir::AffineWriteOpInterface>(op))
      ops.push_back(op);
  });
}

/// Recursively get all the ops belongs to a statement starting from the given
/// operation. The sequence of the operations in defOps will be reversed,
/// depth-first, starting from op. Note that the initial op will be placed in
/// the resulting ops as well.
static void getScopStmtOps(Operation *op, SetVector<Operation *> &ops,
                           SetVector<mlir::Value> &args) {
  // Base case.
  if (!op)
    return;

  // Types of operation that terminates the recusion:
  // Memory allocation ops will be omitted, reaching them means the end of
  // recursion. We will take care of these ops in other passes. The result of
  // these allocation op, i.e., memref, will be
  if (isa<mlir::AllocaOp, mlir::AllocOp>(op)) {
    for (mlir::Value result : op->getResults())
      args.insert(result);
    return;
  }

  // TODO: checks if op has side effects.

  // Keep the op in the given set.
  ops.insert(op);

  // Recursively visit other defining ops that are not in ops.
  for (auto operand : op->getOperands()) {
    // Stop the recursion at block arguments, e.g., loop IVs, external
    // arguments, and insert it into args.
    if (operand.isa<BlockArgument>()) {
      args.insert(operand);
    } else {
      auto defOp = operand.getDefiningOp();
      if (!ops.contains(defOp))
        getScopStmtOps(defOp, ops, args);
    }
  }

  return;
}

static void getCalleeName(unsigned calleeId, CalleeName &calleeName,
                          char prefix = 'S') {
  calleeName.push_back(prefix);
  calleeName += std::to_string(calleeId);
}

/// Create the function definition that contains all the operations that belong
/// to a Scop statement. The function name will be the given calleeName, its
/// contents will be ops, and its type is depend on the given list of args. This
/// callee function has a single block in it, and it has no returned value. The
/// callee will be inserted at the end of the whole module.
static mlir::FuncOp createCallee(StringRef calleeName,
                                 SetVector<Operation *> &ops,
                                 SetVector<mlir::Value> &args, mlir::ModuleOp m,
                                 Operation *writeOp, OpBuilder &b) {
  assert(ops.contains(writeOp) && "writeOp should be a member in ops.");

  unsigned numArgs = args.size();
  unsigned numOps = ops.size();

  // Get a list of types of all function arguments, and use it to create the
  // function type.
  SmallVector<mlir::Type, 8> argTypes;
  for (mlir::Value arg : args)
    argTypes.push_back(arg.getType());
  mlir::FunctionType calleeType = b.getFunctionType(argTypes, llvm::None);

  // Insert the new callee before the end of the module body.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(m.getBody(), std::prev(m.getBody()->end()));

  // Create the callee. Its loc is determined by the writeOp.
  mlir::FuncOp callee =
      b.create<mlir::FuncOp>(writeOp->getLoc(), calleeName, calleeType);
  mlir::Block *entryBlock = callee.addEntryBlock();
  b.setInsertionPointToStart(entryBlock);

  // Create the mapping from the args to the newly created BlockArguments, to
  // replace the uses of the values in the original function to the newly
  // declared entryBlock's input.
  BlockAndValueMapping mapping;
  for (unsigned i = 0; i < numArgs; i++)
    mapping.map(args[i], entryBlock->getArgument(i));

  // Clone the operations into the new callee function. In case they are not in
  // the correct order, we sort them topologically beforehand.
  SetVector<Operation *> sortedOps = topologicalSort(ops);
  for (unsigned i = 0; i < numOps; i++)
    b.clone(*sortedOps[i], mapping);

  // Terminator
  b.create<mlir::ReturnOp>(b.getUnknownLoc());

  return callee;
}

/// Create a caller to the callee right after the writeOp, which will be removed
/// later.
static mlir::CallOp createCaller(mlir::FuncOp callee,
                                 SetVector<mlir::Value> &args,
                                 Operation *writeOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(writeOp);

  return b.create<mlir::CallOp>(writeOp->getLoc(), callee, args.takeVector());
}

/// Remove those ops that are already in the callee, and not have uses by other
/// ops. We will first sort these ops topologically, and then remove them in a
/// reversed order.
static void removeExtractedOps(SetVector<Operation *> &opsToRemove) {
  opsToRemove = topologicalSort(opsToRemove);
  unsigned numOpsToRemove = opsToRemove.size();

  for (unsigned i = 0; i < numOpsToRemove; i++) {
    Operation *op = opsToRemove[numOpsToRemove - i - 1];
    // TODO: need to check if this should be allowed to happen.
    if (op->getUses().empty())
      op->erase();
  }
}

/// The main function that extracts scop statements as functions.
static void extractScopStmt(mlir::FuncOp f, OpBuilder &b) {
  // First discover those write ops that will be the "terminator" of each scop
  // statement in the given function.
  SmallVector<Operation *, 8> writeOps;
  discoverMemWriteOps(f, writeOps);

  unsigned numWriteOps = writeOps.size();

  SetVector<Operation *> ops;
  SetVector<mlir::Value> args;
  SetVector<Operation *> opsToRemove;

  // Use the top-level module to locate places for new functions insertion.
  mlir::ModuleOp m = dyn_cast<mlir::ModuleOp>(f.getParentOp());
  // A writeOp will result in a new caller/callee pair.
  for (unsigned i = 0; i < numWriteOps; i++) {
    ops.clear();
    // Get all the ops inside a statement that corresponds to the current write
    // operation.
    Operation *writeOp = writeOps[i];
    getScopStmtOps(writeOp, ops, args);

    // Get the name of the callee. Should be in the form of "S<id>".
    CalleeName calleeName;
    getCalleeName(i, calleeName);

    // Create the callee.
    mlir::FuncOp callee = createCallee(calleeName, ops, args, m, writeOp, b);
    // Create the caller.
    mlir::CallOp caller = createCaller(callee, args, writeOp, b);

    // All the ops that have been placed in the callee should be removed.
    opsToRemove.set_union(ops);
  }

  // Remove those extracted ops in the original function.
  removeExtractedOps(opsToRemove);
}

namespace {

class ExtractScopStmtPass
    : public mlir::PassWrapper<ExtractScopStmtPass,
                               OperationPass<mlir::FuncOp>> {
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    OpBuilder b(f.getContext());

    extractScopStmt(f, b);
  }
};

} // namespace

void polymer::registerExtractScopStmtPass() {
  PassRegistration<ExtractScopStmtPass>(
      "extract-scop-stmt", "Extract SCoP statements into functions.");
}
