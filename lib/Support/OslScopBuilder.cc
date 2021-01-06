//===- OslScopBuilder.cc ----------------------------------------*- C++ -*-===//
//
// This file implements the class OslScopBuilder that builds OslScop from
// FuncOp.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScopBuilder.h"
#include "polymer/Support/OslScop.h"

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace polymer;
using namespace mlir;
using namespace llvm;

/// Find all CallOp and FuncOp pairs with scop.stmt attribute in the given
/// module and insert them into the OslScop.
static void findAndInsertScopStmts(mlir::FuncOp f,
                                   std::unique_ptr<OslScop> &scop) {
  mlir::ModuleOp m = f.getParentOfType<mlir::ModuleOp>();

  f.walk([&](mlir::CallOp caller) {
    mlir::FuncOp callee = m.lookupSymbol<mlir::FuncOp>(caller.getCallee());

    if (callee.getAttr(OslScop::SCOP_STMT_ATTR_NAME))
      scop->addScopStmt(caller, callee);
  });
}

/// Iterate every value in the given function and annotate them by the symbol in
/// the symbol table of scop, if there is any.
static void setSymbolAttrs(mlir::FuncOp f, std::unique_ptr<OslScop> &scop) {
  // Set symbols for the function BlockArguments.
  llvm::SmallVector<mlir::Attribute, 8> argNames;
  for (unsigned i = 0; i < f.getNumArguments(); i++) {
    mlir::Value arg = f.getArgument(i);
    llvm::StringRef symbol = scop->getSymbol(arg);
    argNames.push_back(StringAttr::get(symbol, f.getContext()));
  }

  f.setAttr(OslScop::SCOP_ARG_NAMES_ATTR_NAME,
            ArrayAttr::get(argNames, f.getContext()));

  // Set symbols for all induction variables in the function.
  f.walk([&](mlir::AffineForOp op) {
    llvm::StringRef symbol = scop->getSymbol(op.getInductionVar());

    op.setAttr(OslScop::SCOP_IV_NAME_ATTR_NAME,
               StringAttr::get(symbol, f.getContext()));
  });

  // Set symbols for all values defined that have a symbol.
  f.walk([&](mlir::Operation *op) {
    llvm::SmallVector<mlir::Attribute, 8> symbols;

    bool hasNonEmptySymbol = false;
    for (mlir::Value result : op->getResults()) {
      llvm::StringRef symbol = scop->getSymbol(result);
      symbols.push_back(StringAttr::get(symbol, f.getContext()));
      if (!symbol.empty())
        hasNonEmptySymbol = true;
    }

    if (hasNonEmptySymbol)
      op->setAttr(OslScop::SCOP_PARAM_NAMES_ATTR_NAME,
                  ArrayAttr::get(symbols, f.getContext()));
  });
}

std::unique_ptr<OslScop> OslScopBuilder::build(mlir::FuncOp f) {
  std::unique_ptr<OslScop> scop = std::make_unique<OslScop>();

  findAndInsertScopStmts(f, scop);

  // If no ScopStmt is constructed, we will discard this build.
  if (scop->getScopStmtMap().size() == 0)
    return nullptr;

  // Now we have put all the ScopStmts into the scop. We can start the
  // initialization then.
  scop->initialize();

  setSymbolAttrs(f, scop);

  return scop;
}
