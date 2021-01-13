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
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "osl-scop"

using namespace polymer;
using namespace mlir;
using namespace llvm;

/// Find all CallOp and FuncOp pairs with scop.stmt attribute in the given
/// module and insert them into the OslScop.
static void findAndInsertScopStmts(mlir::FuncOp f,
                                   OslScopStmtMap &scopStmtMap) {
  mlir::ModuleOp m = f.getParentOfType<mlir::ModuleOp>();

  f.walk([&](mlir::CallOp caller) {
    mlir::FuncOp callee = m.lookupSymbol<mlir::FuncOp>(caller.getCallee());

    if (callee.getAttr(OslScop::SCOP_STMT_ATTR_NAME))
      scopStmtMap.insert(caller, callee);
  });
}

/// Iterate through the ScopStmts in the scopStmtMap and insert them into the
/// ScatTree.
static void initScatTree(ScatTreeNode *root,
                         const OslScopStmtMap &scopStmtMap) {
  for (const auto &it : scopStmtMap) {
    const ScopStmt &scopStmt = it.second;
    root->insertPath(scopStmt.getCaller());
  }
}

/// Initialize symbol table. Parameters and induction variables can be found
/// from the context derived from scopStmtMap.
static OslScopSymbolTable initSymbolTable(const FlatAffineConstraints &ctx,
                                          const OslScopStmtMap &scopStmtMap) {
  OslScopSymbolTable symbolTable;

  llvm::SmallVector<mlir::Value, 8> dimValues, symValues;

  ctx.getIdValues(0, ctx.getNumDimIds(), &dimValues);
  ctx.getIdValues(ctx.getNumDimIds(), ctx.getNumDimAndSymbolIds(), &symValues);

  LLVM_DEBUG(llvm::dbgs() << "Num dim values: " << dimValues.size() << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Num symbol values: " << symValues.size() << "\n");

  // Initialize loop induction variables.
  for (mlir::Value dim : dimValues) {
    assert(dim.isa<mlir::BlockArgument>() &&
           "Values being used as a dim should be a BlockArgument.");
    symbolTable.getOrCreateSymbol(dim);
  }

  // Initialize parameters.
  for (mlir::Value sym : symValues) {
    LLVM_DEBUG(llvm::dbgs() << "Initializing symbol for: " << sym << "\n");
    assert(mlir::isValidSymbol(sym) &&
           "Values being used as a symbol should be valid.");
    symbolTable.getOrCreateSymbol(sym);
  }

  // Initialize all arrays.
  for (const auto &it : scopStmtMap) {
    mlir::CallOp caller = it.second.getCaller();
    LLVM_DEBUG(llvm::dbgs()
               << "Initializing arrays from caller: " << caller << "\n");
    for (mlir::Value arg : caller.getOperands()) {
      LLVM_DEBUG(llvm::dbgs() << arg << " symbol type is: "
                              << symbolTable.getSymbolType(arg));
      if (symbolTable.getSymbolType(arg) == OslScopSymbolTable::MEMREF)
        symbolTable.getOrCreateSymbol(arg);
    }
  }

  return symbolTable;
}

/// Iterate every value in the given function and annotate them by the symbol in
/// the symbol table of scop, if there is any.
static void setSymbolAttrs(mlir::FuncOp f,
                           const OslScopSymbolTable &symbolTable,
                           FlatAffineConstraints ctx) {
  // Set symbols for the function BlockArguments.
  llvm::SmallVector<mlir::Attribute, 8> argNames;
  for (unsigned i = 0; i < f.getNumArguments(); i++) {
    mlir::Value arg = f.getArgument(i);
    llvm::StringRef symbol = symbolTable.getSymbol(arg);
    argNames.push_back(StringAttr::get(symbol, f.getContext()));
  }

  f.setAttr(OslScop::SCOP_ARG_NAMES_ATTR_NAME,
            ArrayAttr::get(argNames, f.getContext()));

  // Set symbols for all induction variables in the function.
  f.walk([&](mlir::AffineForOp op) {
    llvm::StringRef symbol = symbolTable.getSymbol(op.getInductionVar());

    op.setAttr(OslScop::SCOP_IV_NAME_ATTR_NAME,
               StringAttr::get(symbol, f.getContext()));
  });

  // Set symbols for all values defined that have a symbol.
  f.walk([&](mlir::Operation *op) {
    llvm::SmallVector<mlir::Attribute, 8> symbols;

    bool hasNonEmptySymbol = false;
    for (mlir::Value result : op->getResults()) {
      llvm::StringRef symbol = symbolTable.getSymbol(result);
      symbols.push_back(StringAttr::get(symbol, f.getContext()));
      if (!symbol.empty())
        hasNonEmptySymbol = true;
    }

    if (hasNonEmptySymbol)
      op->setAttr(OslScop::SCOP_PARAM_NAMES_ATTR_NAME,
                  ArrayAttr::get(symbols, f.getContext()));
  });

  // Set symbols for the context.
  ctx.projectOut(0, ctx.getNumDimIds());

  IntegerSet iset = ctx.getAsIntegerSet(f.getContext());
  f.setAttr("scop.ctx", IntegerSetAttr::get(iset));

  llvm::SmallVector<mlir::Attribute, 8> ctxParams;
  llvm::SmallVector<mlir::Value, 8> ctxSymValues;
  ctx.getIdValues(ctx.getNumDimIds(), ctx.getNumDimAndSymbolIds(),
                  &ctxSymValues);
  for (mlir::Value ctxSym : ctxSymValues)
    ctxParams.push_back(
        StringAttr::get(symbolTable.getSymbol(ctxSym), f.getContext()));
  f.setAttr("scop.ctx_params", ArrayAttr::get(ctxParams, f.getContext()));
}

std::unique_ptr<OslScop> OslScopBuilder::build(mlir::FuncOp f) {
  std::unique_ptr<OslScop> scop = std::make_unique<OslScop>();

  OslScopStmtMap scopStmtMap;
  ScatTreeNode scatTreeRoot;

  // Initialize and build the scopStmtMap. All the ScopStmts will be extracted
  // from f and inserted into scopStmtMap. Each ScopStmt will have its own
  // domain.
  findAndInsertScopStmts(f, scopStmtMap);

  // If no ScopStmt is constructed, we will discard this build.
  if (scopStmtMap.size() == 0)
    return nullptr;

  // Initialize the ScatTree.

  const FlatAffineConstraints ctx = scopStmtMap.getContext();
  OslScopSymbolTable symbolTable = initSymbolTable(ctx, scopStmtMap);

  setSymbolAttrs(f, symbolTable, ctx);

  return scop;
}
