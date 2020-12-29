//===- OslScopBuilder.cc ----------------------------------------*- C++ -*-===//
//
// This file implements the class OslScopBuilder that builds OslScop from
// FuncOp.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScopBuilder.h"
#include "polymer/Support/OslScop.h"

#include "mlir/IR/BuiltinOps.h"

using namespace polymer;
using namespace mlir;
using namespace llvm;

std::unique_ptr<OslScop> OslScopBuilder::build(mlir::FuncOp f) {
  std::unique_ptr<OslScop> scop = std::make_unique<OslScop>();

  // If no SCopStmt is constructed, we will discard this build.
  if (scop->getScopStmtMap().size() == 0)
    return nullptr;

  return scop;
}
