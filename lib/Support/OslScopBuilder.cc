//===- OslScopBuilder.cc ----------------------------------------*- C++ -*-===//
//
// This file implements the class OslScopBuilder that builds OslScop from
// FuncOp.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScopBuilder.h"
#include "polymer/Support/OslScop.h"

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace polymer;
using namespace mlir;
using namespace llvm;

/// Find all CallOp and FuncOp pairs with scop.stmt attribute in the given
/// module and insert them into the OslScop.
static void buildScopStmtMap(mlir::FuncOp f, std::unique_ptr<OslScop> &scop) {
  mlir::ModuleOp m = f.getParentOfType<mlir::ModuleOp>();

  f.walk([&](mlir::CallOp caller) {
    mlir::FuncOp callee = m.lookupSymbol<mlir::FuncOp>(caller.getCallee());

    if (callee.getAttr(OslScop::SCOP_STMT_ATTR_NAME))
      scop->addScopStmt(caller, callee);
  });
}

std::unique_ptr<OslScop> OslScopBuilder::build(mlir::FuncOp f) {
  std::unique_ptr<OslScop> scop = std::make_unique<OslScop>();

  buildScopStmtMap(f, scop);
  // If no ScopStmt is constructed, we will discard this build.
  if (scop->getScopStmtMap().size() == 0)
    return nullptr;

  FlatAffineConstraints ctx;
  scop->getContextConstraints(ctx);
  scop->initSymbolTable(f, ctx);

  return scop;
}
