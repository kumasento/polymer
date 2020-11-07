//===- Reg2Mem.cc - reg2mem transformation --------------------------------===//
//
// This file implements the reg2mem transformation pass.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/Reg2Mem.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "reg2mem"

static LogicalResult createMemRefFromDomain(FlatAffineConstraints &cst,
                                            mlir::Value &memref,
                                            mlir::Type valType,
                                            mlir::OpBuilder &b,
                                            int64_t dimSize = 32) {
  if (cst.getNumDimIds() == 0) {
    // If there is no dim, we create a scalar memory.
    memref = b.create<mlir::AllocOp>(b.getUnknownLoc(),
                                     MemRefType::get({1}, valType));
  } else {
    // If not, we create a memory based on the upper bounds of the dims.
    assert(!cst.isEmpty() &&
           "There should exist constraints when there are dims.");

    // For now, we create a memref of a fixed size.
    // TODO: we should get bounds from the constraints.
    unsigned rank = cst.getNumDimIds();

    std::vector<int64_t> dimSizes(rank, dimSize);
    memref = b.create<mlir::AllocOp>(b.getUnknownLoc(),
                                     MemRefType::get(dimSizes, valType));
  }

  return success();
}

static LogicalResult demoteRegisterToMemory(mlir::FuncOp f,
                                            mlir::OpBuilder &b) {
  // Collect the liveness information for all blocks in the current function.
  Liveness liveness(f);

  // Extract all affine.for operations.
  llvm::SmallVector<AffineForOp, 4> forOps;
  f.walk([&](mlir::AffineForOp forOp) { forOps.push_back(forOp); });

  // For each affine.for, we check their live-in values. For each of them, if
  // it is a register, we go to where that value is defined, store that value
  // in a scratchpad memory, and load it into the current block at its
  // beginning.
  for (auto forOp : forOps) {
    auto &allInValues = liveness.getLiveIn(forOp.getBody());

    for (auto inVal : allInValues) {
      // We ignore all the block arguments. They are loop IVs.
      if (inVal.dyn_cast<mlir::BlockArgument>())
        continue;

      auto defOp = inVal.getDefiningOp();

      // We ignore all the values defined by:
      // - AllocOp: memref itself cannot be demoted
      // - DimOp: the dimensionality will be handled by other passes.
      if (isa<mlir::AllocOp, mlir::DimOp>(defOp))
        continue;

      // TODO: mark as DEBUG
      LLVM_DEBUG({ inVal.dump(); });

      // Extract the domain of the defining op for inVal.
      llvm::SmallVector<Operation *, 4> enclosingOps;
      mlir::getEnclosingAffineForAndIfOps(*defOp, &enclosingOps);
      LLVM_DEBUG({
        llvm::dbgs() << "Number of enclosing affine for & if: "
                     << enclosingOps.size() << "\n";
      });

      FlatAffineConstraints cst;
      if (failed(mlir::getIndexSet(enclosingOps, &cst)))
        return failure();

      LLVM_DEBUG({
        llvm::dbgs() << "Domain constraints for the defining op:\n";
        cst.dump();
      });

      // Create an individual memref for each value.
      // TODO: there should be a way to minimise the number of memrefs
      // created.
      mlir::Value scratchpad;
      const int64_t scratchpadSizePerDim = 32;

      // Insert the scratchpad right at the start of the function.
      // TODO: should change the location if we need to refer symbols of SSA
      // values.
      auto &entryBlock = *f.getBlocks().begin();
      b.setInsertionPointToStart(&entryBlock);

      if (failed(createMemRefFromDomain(cst, scratchpad, inVal.getType(), b,
                                        /*dimSize=*/scratchpadSizePerDim)))
        return failure();

      MemRefType memTy = scratchpad.getType().dyn_cast<MemRefType>();
      int64_t rank = memTy.getRank();

      // Get the address for load and store scratchpad values.
      AffineMap addrMap;
      // If the memref created is rank 1 and has a single element, we could just
      // use '0' as its access address. Otherwise, we use the newly calculated
      // addrExprs.
      if (rank == 1 && memTy.getDimSize(0) == 1) {
        addrMap =
            AffineMap::get(cst.getNumDimIds(), 0, b.getAffineConstantExpr(0));
      } else {
        llvm::SmallVector<AffineExpr, 8> addrExprs;
        // The address to access the scratchpad is calculated by the current IV
        // mod by the pre-known, fixed scratchpad size.
        for (int64_t i = 0; i < rank; i++)
          addrExprs.push_back(b.getAffineDimExpr(i) % scratchpadSizePerDim);

        addrMap =
            AffineMap::get(cst.getNumDimIds(), 0, addrExprs, b.getContext());
      }

      // Get the indices of all the enclosing for-loops that will be applied for
      // address calculation.
      llvm::SmallVector<mlir::Value, 8> addrInds;
      for (auto op : enclosingOps)
        if (auto enclosingForOp = dyn_cast<mlir::AffineForOp>(op))
          addrInds.push_back(enclosingForOp.getInductionVar());

      // Store the live-in value into the scratchpad.
      b.setInsertionPointAfter(defOp);
      b.create<mlir::AffineStoreOp>(b.getUnknownLoc(), inVal, scratchpad,
                                    addrMap, addrInds);

      // Load the live-in value back from the scratchpad.
      b.setInsertionPointToStart(forOp.getBody());
      auto newInVal = b.create<mlir::AffineLoadOp>(
          b.getUnknownLoc(), scratchpad, addrMap, addrInds);

      // Replace every use of the live-in value by the result of the new load
      // operation. Note that we should only replace those in the current block.
      inVal.replaceUsesWithIf(newInVal, [&](mlir::OpOperand &operand) -> bool {
        return operand.getOwner()->getBlock() == forOp.getBody();
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
