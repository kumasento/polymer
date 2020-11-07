//===- PlutoTransform.cc - Transform MLIR code by PLUTO -------------------===//
//
// This file implements the transformation passes on MLIR using PLUTO.
//
//===----------------------------------------------------------------------===//

#include "polymer/Transforms/PlutoTransform.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopStmtOpSet.h"
#include "polymer/Support/OslSymbolTable.h"
#include "polymer/Target/OpenScop.h"

#include "pluto/internal/pluto.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
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
using namespace polymer;

/// Insert value mapping into the given mapping object based on the provided src
/// and dst symbol tables.
static LogicalResult updateValueMapping(OslSymbolTable &srcTable,
                                        OslSymbolTable &dstTable,
                                        BlockAndValueMapping &mapping) {
  // TODO: check the symbol compatibility between srcTable and dstTable.
  SmallVector<StringRef, 8> symbols;
  srcTable.getValueSymbols(symbols);

  for (auto sym : symbols) {
    if (auto dstVal = dstTable.getValue(sym))
      mapping.map(srcTable.getValue(sym), dstVal);
    else {
      llvm::errs()
          << "Symbol " << sym
          << " in the source table is not found in the destination table.\n";
      return failure();
    }
  }
  return success();
}

static LogicalResult plutoTilingOpt(mlir::FuncOp funcOp, OpBuilder &b) {
  PlutoContext *context = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;

  std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(funcOp, srcTable);
  if (!scop) {
    funcOp.emitError(
        "Cannot emit a valid OpenScop representation from the given FuncOp.");
    return failure();
  }

  if (scop->getNumStatements() == 0) {
    return success();
  }

  // Should use isldep, candl cannot work well for this case.
  // TODO: should discover why.
  context->options->isldep = 1;

  PlutoProg *prog = osl_scop_to_pluto_prog(scop->get(), context);

  pluto_compute_dep_directions(prog);
  pluto_compute_dep_satisfaction(prog);
  pluto_tile(prog);

  pluto_populate_scop(scop->get(), prog, context);
  // osl_scop_print(stdout, scop->get());

  auto moduleOp = dyn_cast<mlir::ModuleOp>(funcOp.getParentOp());

  // TODO: remove the root update pairs.
  auto newFuncOp = createFuncOpFromOpenScop(std::move(scop), moduleOp, dstTable,
                                            b.getContext());

  BlockAndValueMapping mapping;
  // TODO: refactorize this function and the following logic.
  if (failed(updateValueMapping(srcTable, dstTable, mapping)))
    return failure();

  SmallVector<StringRef, 8> stmtSymbols;
  srcTable.getOpSetSymbols(stmtSymbols);
  for (auto stmtSym : stmtSymbols) {
    // The operation to be cloned.
    auto srcOpSet = srcTable.getOpSet(stmtSym);
    // The clone destination.
    auto dstOpSet = dstTable.getOpSet(stmtSym);

    // There should be 2 * N number of dst ops, ordered as pairs of <caller,
    // callee>. We should replace each caller by the contents of srcOpSet and
    // erase the placeholder callee.
    unsigned dstOpSetSize = dstOpSet.size();
    assert(dstOpSetSize % 2 == 0 &&
           "There should be even number of caller/callee.");

    for (unsigned i = 0; i < dstOpSetSize; i += 2) {
      auto caller = dstOpSet.get(i);

      // Update the mapping based on the caller args.
      for (auto operand : caller->getOperands()) {
        if (dstTable.ivArgToName.find(operand) != dstTable.ivArgToName.end()) {
          auto sym = dstTable.ivArgToName[operand];
          auto prefix = sym.substr(0, sym.find('_'));

          if (!prefix.empty()) {
            auto srcArg = srcTable.getValue(dstTable.scatNameToIter[prefix]);
            assert(srcArg != nullptr);

            mapping.map(srcArg, operand);
          }
        }
      }

      b.setInsertionPoint(caller);

      for (unsigned j = 0, e = srcOpSet.size(); j < e; j++)
        b.clone(*(srcOpSet.get(e - j - 1)), mapping);

      caller->erase();
      dstOpSet.get(i + 1)->erase();
    }
  }

  pluto_context_free(context);

  funcOp.erase();

  return success();
}

namespace {
/// TODO: split this into specific categories like tiling.
class PlutoTransformPass
    : public mlir::PassWrapper<PlutoTransformPass,
                               OperationPass<mlir::FuncOp>> {
public:
  void runOnOperation() override {
    mlir::FuncOp f = getOperation();
    auto builder = OpBuilder(f.getContext());

    if (failed(plutoTilingOpt(f, builder)))
      signalPassFailure();
  }
};

} // namespace

void polymer::registerPlutoTransformPass() {
  PassRegistration<PlutoTransformPass>("pluto-opt",
                                       "Optimization implemented by PLUTO.");
}
