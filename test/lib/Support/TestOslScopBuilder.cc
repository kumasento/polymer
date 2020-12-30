//===- TestOslScopBuilder.cc ------------------------------------*- C++ -*-===//
//
// This file implements the test passes for the OslScopBuilder.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopBuilder.h"

#include <string>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

namespace {
struct TestOslScopBuilderPass
    : public PassWrapper<TestOslScopBuilderPass, FunctionPass> {
  void runOnFunction() override {
    FuncOp f = getFunction();
    if (f.getAttr(OslScop::SCOP_STMT_ATTR_NAME))
      return;

    std::unique_ptr<OslScop> scop = OslScopBuilder().build(f);
    f.emitRemark("Has OslScop: ") << llvm::toStringRef(scop != nullptr);

    if (scop == nullptr)
      return;

    // Check the content of the context after projecting out all dim values.
    FlatAffineConstraints ctx;
    scop->getContextConstraints(ctx);
    ctx.projectOut(0, ctx.getNumDimIds());

    IntegerSet iset = ctx.getAsIntegerSet(f.getContext());
    f.setAttr("scop.ctx", IntegerSetAttr::get(iset));

    llvm::SmallVector<mlir::Attribute, 8> ctxParams;
    llvm::SmallVector<mlir::Value, 8> ctxSymValues;
    ctx.getIdValues(ctx.getNumDimIds(), ctx.getNumDimAndSymbolIds(),
                    &ctxSymValues);
    for (mlir::Value ctxSym : ctxSymValues)
      ctxParams.push_back(
          StringAttr::get(scop->getOrCreateSymbol(ctxSym), f.getContext()));
    f.setAttr("scop.ctx_params", ArrayAttr::get(ctxParams, f.getContext()));
  }
};
} // namespace

namespace polymer {

void registerTestOslScopBuilder() {
  PassRegistration<TestOslScopBuilderPass>(
      "test-osl-scop-builder", "Print the built OslScop of a function.");
}
} // namespace polymer
