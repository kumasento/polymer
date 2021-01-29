//===- ExtractScopStmt.cc - Extract scop stmt to func -----------------C++-===//
//
// This file implements the transformation that extracts scop statements into
// MLIR functions.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/ExtractScopStmt.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
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
/// TODO: support CallOp.
static void discoverMemWriteOps(mlir::FuncOp f,
                                SmallVectorImpl<Operation *> &ops) {
  f.getOperation()->walk([&](Operation *op) {
    if (isa<mlir::AffineWriteOpInterface>(op))
      ops.push_back(op);
  });
}

/// Go through every write op and get their domain. And we try to union all
/// their parameters.
static void getGlobalContext(ArrayRef<Operation *> ops,
                             SetVector<Value> &params) {
  params.clear();

  for (Operation *op : ops) {
    FlatAffineConstraints cst;
    SmallVector<Operation *, 8> enclosingOps;
    getEnclosingAffineForAndIfOps(*op, &enclosingOps);
    getIndexSet(enclosingOps, &cst);

    SmallVector<Value, 8> symbols;
    cst.getIdValues(cst.getNumDimIds(), cst.getNumDimAndSymbolIds(), &symbols);

    for (Value symbol : symbols)
      params.insert(symbol);
  }
}

/// Get all the ops belongs to a statement starting from the given
/// operation. The sequence of the operations in defOps will be reversed,
/// depth-first, starting from op. Note that the initial op will be placed in
/// the resulting ops as well.
static void getScopStmtOps(Operation *writeOp, SetVector<Operation *> &ops,
                           SetVector<Value> &params,
                           SetVector<mlir::Value> &args) {
  SmallVector<Operation *, 8> worklist;
  worklist.push_back(writeOp);
  ops.insert(writeOp);

  // Resolve the domain of the writeOp.
  FlatAffineConstraints cst;
  SmallVector<Operation *, 8> enclosingOps;
  getEnclosingAffineForAndIfOps(*writeOp, &enclosingOps);
  getIndexSet(enclosingOps, &cst);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // Types of operation that terminates the recusion:
    // Memory allocation ops will be omitted, reaching them means the end of
    // recursion. We will take care of these ops in other passes. The result of
    // these allocation op, i.e., memref, will be treated as input arguments to
    // the new statement function.
    if (isa<mlir::AllocaOp, mlir::AllocOp>(op)) {
      for (mlir::Value result : op->getResults())
        args.insert(result);
      continue;
    }

    // Keep the op in the given set. ops also stores the "visited" information:
    // any op inside ops will be treated as visited and won't be inserted into
    // the worklist again.
    ops.insert(op);

    // Recursively visit other defining ops that are not in ops.
    for (mlir::Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();

      // Check whether the operand is of the domain or the global parameter set.
      unsigned pos;
      bool isOperandOfDomain =
          cst.findId(operand, &pos) || params.contains(operand);

      if (defOp && !isOperandOfDomain) {
        // We find the defining op and place it in the worklist, if it is not
        // null, not an operand of the domain, and has not been visited yet.
        if (!ops.contains(defOp)) // Don't nest this with other conditions.
          worklist.push_back(defOp);
      } else {
        // Otherwise, stop the recursion at values that don't have a defining
        // op, i.e., block arguments, which could be loop IVs, external
        // arguments, etc. And insert them into the argument list (args).
        args.insert(operand);
      }
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
/// contents will be ops, and its type depends on the given list of args. This
/// callee function has a single block in it, and it has no returned value. The
/// callee will be inserted at the end of the whole module.
static mlir::FuncOp createCallee(StringRef calleeName,
                                 const SetVector<Operation *> &ops,
                                 const SetVector<mlir::Value> &args,
                                 mlir::ModuleOp m, Operation *writeOp,
                                 OpBuilder &b) {
  assert(ops.contains(writeOp) && "writeOp should be a member in ops.");

  unsigned numOps = ops.size();

  // Get a list of types of all function arguments, and use it to create the
  // function type.
  TypeRange argTypes = ValueRange(args.getArrayRef()).getTypes();
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
  mapping.map(args, entryBlock->getArguments());

  // Clone the operations into the new callee function. In case they are not in
  // the correct order, we sort them topologically beforehand.
  SetVector<Operation *> sortedOps = topologicalSort(ops);
  for (unsigned i = 0; i < numOps; i++)
    b.clone(*sortedOps[i], mapping);

  // Terminator
  b.create<mlir::ReturnOp>(callee.getLoc());
  // Set the scop_stmt attribute for identification at a later stage.
  // TODO: in the future maybe we could create a customized dialect, e.g., Scop,
  // that contains scop stmt FuncOp, e.g., ScopStmtOp.
  callee.setAttr("scop.stmt", b.getUnitAttr());

  return callee;
}

/// Create a caller to the callee right after the writeOp, which will be removed
/// later.
static mlir::CallOp createCaller(mlir::FuncOp callee,
                                 const SetVector<mlir::Value> &args,
                                 Operation *writeOp, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(writeOp);

  return b.create<mlir::CallOp>(writeOp->getLoc(), callee,
                                ValueRange(args.getArrayRef()));
}

/// Remove those ops that are already in the callee (given by `opsToRemove`),
/// and not have uses by other ops (need to be checked by getUses().empty()). We
/// will first sort these ops topologically, and then remove them in a reversed
/// order. This is like DCE on a subset of operations.
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

/// The main function that extracts scop statements as functions. Returns the
/// number of callees extracted from this function.
static unsigned extractScopStmt(mlir::FuncOp f, unsigned numCallees,
                                OpBuilder &b) {
  // First discover those write ops that will be the "terminator" of each scop
  // statement in the given function.
  SmallVector<Operation *, 8> writeOps;
  discoverMemWriteOps(f, writeOps);

  SetVector<Value> globalParams;
  getGlobalContext(writeOps, globalParams);

  SetVector<Operation *> opsToRemove;

  unsigned numWriteOps = writeOps.size();

  // Use the top-level module to locate places for new functions insertion.
  mlir::ModuleOp m = cast<mlir::ModuleOp>(f.getParentOp());
  // A writeOp will result in a new caller/callee pair.
  for (unsigned i = 0; i < numWriteOps; i++) {
    SetVector<Operation *> ops;
    SetVector<mlir::Value> args;

    // Get all the ops inside a statement that corresponds to the current write
    // operation.
    Operation *writeOp = writeOps[i];
    getScopStmtOps(writeOp, ops, globalParams, args);

    // Get the name of the callee. Should be in the form of "S<id>".
    CalleeName calleeName;
    getCalleeName(i + numCallees, calleeName);

    // Create the callee.
    mlir::FuncOp callee = createCallee(calleeName, ops, args, m, writeOp, b);
    // Create the caller.
    createCaller(callee, args, writeOp, b);

    // All the ops that have been placed in the callee should be removed.
    opsToRemove.set_union(ops);
  }

  // Remove those extracted ops in the original function.
  removeExtractedOps(opsToRemove);

  return numWriteOps;
}

namespace {

class ExtractScopStmtPass
    : public mlir::PassWrapper<ExtractScopStmtPass,
                               OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    SmallVector<mlir::FuncOp, 4> funcs;
    m.walk([&](mlir::FuncOp f) { funcs.push_back(f); });

    unsigned numCallees = 0;
    for (mlir::FuncOp f : funcs)
      numCallees += extractScopStmt(f, numCallees, b);
  }
};

} // namespace

void polymer::registerExtractScopStmtPass() {
  PassRegistration<ExtractScopStmtPass>(
      "extract-scop-stmt", "Extract SCoP statements into functions.");
}
