//===- OslScopBuilder.cc ----------------------------------------*- C++ -*-===//
//
// This file implements the class OslScopBuilder that builds OslScop from
// FuncOp.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScopBuilder.h"
#include "polymer/Support/OslScop.h"

#include "mlir/Analysis/AffineAnalysis.h"
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

  // "walk" ensures that the insertion order is aligned with the execution
  // order of statements.
  f.walk([&](mlir::CallOp caller) {
    mlir::FuncOp callee = m.lookupSymbol<mlir::FuncOp>(caller.getCallee());

    if (callee.getAttr(OslScop::SCOP_STMT_ATTR_NAME))
      scopStmtMap.insert(caller, callee);
  });
}

/// Iterate through the ScopStmts in the scopStmtMap and insert them into the
/// ScatTree.
static void initScatTree(ScatTreeNode &root, const OslScopStmtMap &sMap) {
  for (const auto &key : sMap.getKeys()) {
    const ScopStmt &scopStmt = sMap.lookup(key);
    scopStmt.updateScatTree(root);
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
    symbolTable.lookupOrCreate(dim);
  }

  // Initialize parameters.
  for (mlir::Value sym : symValues) {
    LLVM_DEBUG(llvm::dbgs() << "Initializing symbol for: " << sym << "\n");
    assert(mlir::isValidSymbol(sym) &&
           "Values being used as a symbol should be valid.");
    symbolTable.lookupOrCreate(sym);
  }

  // Initialize all arrays.
  for (const auto &it : scopStmtMap) {
    mlir::CallOp caller = it.second.getCaller();
    LLVM_DEBUG(llvm::dbgs()
               << "Initializing arrays from caller: " << caller << "\n");
    for (mlir::Value arg : caller.getOperands()) {
      LLVM_DEBUG(llvm::dbgs()
                 << arg << " symbol type is: " << symbolTable.getType(arg));
      if (symbolTable.getType(arg) == OslScopSymbolTable::MEMREF)
        symbolTable.lookupOrCreate(arg);
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
    llvm::StringRef symbol = symbolTable.lookup(arg);
    argNames.push_back(StringAttr::get(symbol, f.getContext()));
  }

  f.setAttr(OslScop::SCOP_ARG_NAMES_ATTR_NAME,
            ArrayAttr::get(argNames, f.getContext()));

  // Set symbols for all induction variables in the function.
  f.walk([&](mlir::AffineForOp op) {
    llvm::StringRef symbol = symbolTable.lookup(op.getInductionVar());

    op.setAttr(OslScop::SCOP_IV_NAME_ATTR_NAME,
               StringAttr::get(symbol, f.getContext()));
  });

  // Set symbols for all values defined that have a symbol.
  f.walk([&](mlir::Operation *op) {
    llvm::SmallVector<mlir::Attribute, 8> symbols;

    bool hasNonEmptySymbol = false;
    for (mlir::Value result : op->getResults()) {
      llvm::StringRef symbol = symbolTable.lookup(result);
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
        StringAttr::get(symbolTable.lookup(ctxSym), f.getContext()));
  f.setAttr("scop.ctx_params", ArrayAttr::get(ctxParams, f.getContext()));
}

static void setScatTreeAttrs(const OslScopStmtMap &sMap,
                             const ScatTreeNode &root) {
  for (const auto &key : sMap.getKeys()) {
    const ScopStmt &s = sMap.lookup(key);
    llvm::SmallVector<unsigned, 8> scats;
    s.getScats(root, scats);

    mlir::CallOp caller = s.getCaller();

    llvm::SmallVector<mlir::Attribute, 8> scatsAttr;
    for (unsigned id : scats)
      scatsAttr.push_back(
          IntegerAttr::get(IntegerType::get(64, caller.getContext()), id));
    caller.setAttr(OslScop::SCOP_STMT_SCATS_NAME,
                   ArrayAttr::get(scatsAttr, caller.getContext()));
  }
}

/// Set the attribute that represents the access constraints for each operation
/// in each ScopStmt.
static void setAccessAttrs(const OslScopStmtMap &sm,
                           const OslScopSymbolTable &st,
                           const FlatAffineConstraints &ctx) {
  for (const auto &key : sm.getKeys()) {
    const ScopStmt &s = sm.lookup(key);
    llvm::SmallVector<mlir::MemRefAccess, 8> acs;
    s.getAccesses(acs);

    for (const mlir::MemRefAccess &ac : acs) {
      FlatAffineConstraints cst;
      s.getAccessConstraints(ac, st, cst);

      mlir::Operation *op = ac.opInst;
      IntegerSet iset = cst.getAsIntegerSet(op->getContext());
      op->setAttr(OslScop::SCOP_STMT_ACCESS_NAME, IntegerSetAttr::get(iset));

      llvm::SmallVector<mlir::Attribute, 8> symbolAttrs;
      for (unsigned i = 0; i < cst.getNumDimAndSymbolIds(); i++) {
        Optional<mlir::Value> id = cst.getId(i);
        if (id.hasValue()) {
          mlir::Value value = id.getValue();
          symbolAttrs.push_back(
              StringAttr::get(st.lookup(value), op->getContext()));
        } else {
          symbolAttrs.push_back(StringAttr::get("", op->getContext()));
        }
      }

      op->setAttr(OslScop::SCOP_STMT_ACCESS_SYMBOLS_NAME,
                  ArrayAttr::get(symbolAttrs, op->getContext()));
    }
  }
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
  initScatTree(scatTreeRoot, scopStmtMap);

  // Initialize the context and the symbol table.
  const FlatAffineConstraints ctx = scopStmtMap.getContext();
  OslScopSymbolTable symbolTable = initSymbolTable(ctx, scopStmtMap);

  setSymbolAttrs(f, symbolTable, ctx);
  setScatTreeAttrs(scopStmtMap, scatTreeRoot);
  setAccessAttrs(scopStmtMap, symbolTable, ctx);

  return scop;
}
