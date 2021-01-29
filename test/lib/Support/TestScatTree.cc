//===- TestScatTree.cc ------------------------------------------*- C++ -*-===//
//
// This file implements the test passes for the scattering tree utilities.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/ScatTree.h"

#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace polymer;

static std::string getScatsString(llvm::SmallVectorImpl<unsigned> &scats) {
  std::string result;

  result = "{ ";

  for (unsigned i = 0; i < scats.size(); i++) {
    result += std::to_string(scats[i]);
    if (i != scats.size() - 1)
      result += ", ";
  }

  result += " }";

  return result;
}

namespace {
struct TestScatTreePass : PassWrapper<TestScatTreePass, FunctionPass> {
  void runOnFunction() override {
    FuncOp f = getFunction();

    ScatTreeNode root;
    f.getBody().walk([&](Operation *op) {
      if (isa<mlir::AffineForOp, mlir::AffineIfOp, mlir::AffineYieldOp>(op))
        return;

      llvm::SmallVector<unsigned, 8> scats;
      root.insertPath(op, scats);

      op->emitRemark("Scats: ") << getScatsString(scats);
    });

    f->emitRemark("Tree depth: ") << root.getDepth();
  }
};
} // namespace

namespace polymer {

void registerTestScatTree() {
  PassRegistration<TestScatTreePass>(
      "test-scat-tree",
      "Print the scattering of each statement in enclosed affine.for");
}

} // namespace polymer
