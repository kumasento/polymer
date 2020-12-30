//===- TestOslScopBuilder.cc ------------------------------------*- C++ -*-===//
//
// This file implements the test passes for the OslScopBuilder.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopBuilder.h"

#include <string>

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

static std::string
getFlatAffineConstraintsAsString(const FlatAffineConstraints &cst) {
  std::string s;

  s += "eqs: {";
  for (unsigned i = 0; i < cst.getNumEqualities(); i++) {
    s += "{ ";
    for (unsigned j = 0; j < cst.getNumCols(); j++) {
      s += std::to_string(cst.atEq(i, j));
      if (j < cst.getNumCols() - 1)
        s += ", ";
    }
    s += " }";
    if (i < cst.getNumEqualities() - 1)
      s += ", ";
  }

  s += "} inEqs: {";
  for (unsigned i = 0; i < cst.getNumInequalities(); i++) {
    s += "{ ";
    for (unsigned j = 0; j < cst.getNumCols(); j++) {
      s += std::to_string(cst.atIneq(i, j));
      if (j < cst.getNumCols() - 1)
        s += ", ";
    }
    s += " }";
    if (i < cst.getNumInequalities() - 1)
      s += ", ";
  }
  s += "}";
  return s;
}

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

    FlatAffineConstraints ctx;
    scop->getContextConstraints(ctx);
    f.emitRemark("Num context parameters: ") << ctx.getNumDimAndSymbolIds();
    f.emitRemark() << getFlatAffineConstraintsAsString(ctx);
  }
};
} // namespace

namespace polymer {

void registerTestOslScopBuilder() {
  PassRegistration<TestOslScopBuilderPass>(
      "test-osl-scop-builder", "Print the built OslScop of a function.");
}
} // namespace polymer
