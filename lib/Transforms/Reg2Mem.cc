//===- Reg2Mem.cc - reg2mem transformation --------------------------------===//
//
// This file implements the reg2mem transformation pass.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/Reg2Mem.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "reg2mem"

using DefToUsesMap =
    llvm::DenseMap<mlir::Value, llvm::SetVector<mlir::Operation *>>;

/// Build the mapping of values from where they are defined to where they are
/// used. We will need this information to decide whether a Value should be
/// stored in a scratchpad, and if so, what the scratchpad should look like.
/// Note that we only care about those values that are on the use-def chain that
/// ends up with an affine.store operation. We also ignore all the def-use pairs
/// that are in the same block.
static LogicalResult mapDefToUses(mlir::FuncOp f, DefToUsesMap &defToUses) {
  f.walk([&](mlir::AffineStoreOp storeOp) {
    // Assuming the def-use chain is acyclic.
    llvm::SmallVector<mlir::Operation *, 8> ops;
    ops.push_back(storeOp);

    while (!ops.empty()) {
      mlir::Operation *op = ops.back();
      ops.pop_back();

      for (mlir::Value v : op->getOperands()) {
        mlir::Operation *defOp = v.getDefiningOp();
        // Don't need to go further if v is defined by the following operations.
        // TODO: how to handle the case that v is a BlockArgument, i.e., loop
        // carried values?
        if (!defOp || isa<mlir::AllocOp, mlir::DimOp>(defOp))
          continue;

        // The block that defines the value is different from the block of the
        // current op.
        if (v.getParentBlock() != op->getBlock()) {
          if (defToUses.find(v) == defToUses.end())
            defToUses[v] = {};
          defToUses[v].insert(op);
        }

        // No need to look at the operands of the following list of operations.
        if (!isa<mlir::AffineLoadOp>(defOp))
          ops.push_back(defOp);
      }
    }
  });

  return success();
}

/// Creates a single-entry scratchpad memory that stores values from the
/// defining point and can be loaded when needed at the uses.
static LogicalResult createScratchpadAllocaOp(mlir::Value val,
                                              mlir::AllocaOp &allocaOp,
                                              mlir::OpBuilder &b) {
  // Sanity checks on the defining op.
  mlir::Operation *defOp = val.getDefiningOp();
  assert(defOp && "val should have a valid defining operation.");

  // Set the allocation point after where the val is defined.
  b.setInsertionPointAfter(defOp);

  // The memref shape is 1 and the type is derived from val.
  allocaOp = b.create<mlir::AllocaOp>(defOp->getLoc(),
                                      MemRefType::get({1}, val.getType()));

  return success();
}

/// Creata an AffineStoreOp for the value to be stored on the scratchpad.
static LogicalResult createScratchpadStoreOp(mlir::Value valToStore,
                                             mlir::AllocaOp allocaOp,
                                             mlir::AffineStoreOp &storeOp,
                                             mlir::OpBuilder &b) {
  // Create a storeOp to the memref using address 0. The new storeOp will be
  // placed right after the allocaOp, and its location is hinted by allocaOp.
  // Here we assume that allocaOp is dominated by the defining op of valToStore.
  b.setInsertionPointAfter(allocaOp);
  storeOp = b.create<mlir::AffineStoreOp>(
      allocaOp.getLoc(), valToStore, allocaOp.getResult(),
      b.getConstantAffineMap(0), std::vector<mlir::Value>());

  return success();
}

/// Create an AffineLoadOp for the value stored in the scratchpad. The insertion
/// point will be at the beginning of the block of the useOp, such that all the
/// subsequent uses of the Value in the scratchpad can re-use the same load
/// result. Note that we don't check whether the useOp is still using the
/// original value that is stored in the scratchpad (some replacement could
/// happen already), you need to do that before calling this function to avoid
/// possible redundancy. This function won't replace uses.
static LogicalResult createScratchpadLoadOp(mlir::AllocaOp allocaOp,
                                            mlir::Operation *useOp,
                                            mlir::AffineLoadOp &loadOp,
                                            mlir::OpBuilder &b) {
  // The insertion point will be at the beginning of the parent block for useOp.
  b.setInsertionPointToStart(useOp->getBlock());
  // The location is set to be the useOp that will finally use this newly
  // created load op. The address is set to be 0 since the memory has only one
  // element in it. You will need to replace the input to useOp outside.
  loadOp = b.create<mlir::AffineLoadOp>(useOp->getLoc(), allocaOp.getResult(),
                                        b.getConstantAffineMap(0),
                                        std::vector<mlir::Value>());

  return success();
}

static LogicalResult demoteRegisterToMemory(mlir::FuncOp f, OpBuilder &b) {
  // Get the mapping from a value to its uses that are in a different block as
  // where the value itself is defined.
  DefToUsesMap defToUses;
  if (failed(mapDefToUses(f, defToUses)))
    return failure();

  // Handle each def-use pair in in the current function.
  for (auto defUsesPair : defToUses) {
    // The value to be stored in a scratchpad.
    mlir::Value val = defUsesPair.first;

    // Create the alloca op for the scratchpad.
    mlir::AllocaOp allocaOp;
    if (failed(createScratchpadAllocaOp(val, allocaOp, b)))
      return failure();

    // Create the store op that stores val into the scratchpad for future uses.
    mlir::AffineStoreOp storeOp;
    if (failed(createScratchpadStoreOp(val, allocaOp, storeOp, b)))
      return failure();

    // Iterate each use of val, and create the load op, the result of which will
    // replace the original val. After creating this load op, we replaces the
    // uses of the original val in the same block as the load op by the result
    // of it. And for those already replaced, we pop them out of the list to be
    // processed (useOps).
    llvm::SetVector<mlir::Operation *> useOps = defUsesPair.second;

    while (!useOps.empty()) {
      // Get the useOp to be processed from the back of the set.
      mlir::Operation *useOp = useOps.pop_back_val();

      // Create the load op for it.
      mlir::AffineLoadOp loadOp;
      if (failed(createScratchpadLoadOp(allocaOp, useOp, loadOp, b)))
        return failure();

      // Replace the uses of val in the same block as useOp (or loadOp). And if
      // such a use is found, we will remove them from the useOps set if they
      // are still in it.
      val.replaceUsesWithIf(loadOp.getResult(), [&](mlir::OpOperand &operand) {
        mlir::Operation *currUseOp = operand.getOwner();

        //  Check the equivalence of the blocks.
        if (currUseOp->getBlock() == useOp->getBlock()) {
          // Remove currUseOp if it is still in useOps.
          if (useOps.count(currUseOp) != 0)
            useOps.remove(currUseOp);

          return true;
        }

        return false;
      });
    }
  }

  return success();
}

namespace {

class RegToMemPass
    : public mlir::PassWrapper<RegToMemPass, OperationPass<mlir::FuncOp>> {

public:
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    auto builder = OpBuilder(f.getContext());

    if (failed(demoteRegisterToMemory(f, builder)))
      signalPassFailure();
  }
};

} // namespace

void polymer::registerRegToMemPass() {
  PassRegistration<RegToMemPass>("reg2mem", "Demote register to memref.");
}
