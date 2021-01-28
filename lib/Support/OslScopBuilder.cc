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
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
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

/// Update the context constraints to make them compatible with the context
/// relation requirements by OpenScop.
static FlatAffineConstraints getOslContext(FlatAffineConstraints ctx) {
  FlatAffineConstraints cst;
  cst.mergeAndAlignIdsWithOther(0, &ctx);

  // Strip all the IDs that are not symbols.
  while (cst.getNumDimIds() > 0)
    cst.removeId(0);
  while (cst.getNumLocalIds() > 0)
    cst.removeId(cst.getNumDimAndSymbolIds());
  assert(cst.getNumSymbolIds() == cst.getNumIds());

  // Insert constraints from the ctx that are not related to dims or locals.
  for (unsigned i = 0; i < ctx.getNumConstraints(); i++) {
    bool isEq = i < ctx.getNumEqualities();
    llvm::ArrayRef<int64_t> row =
        isEq ? ctx.getEquality(i)
             : ctx.getInequality(i - ctx.getNumEqualities());

    // If all dim or local values in the current row are zero, append it to cst.
    if (llvm::all_of(row.take_front(ctx.getNumDimIds()),
                     [](const int64_t v) { return v == 0; }) &&
        llvm::all_of(row.take_back(ctx.getNumLocalIds() + 1).drop_back(),
                     [](const int64_t v) { return v == 0; })) {
      llvm::ArrayRef<int64_t> symInRow =
          row.drop_front(ctx.getNumDimIds()).take_front(ctx.getNumSymbolIds());

      // Create a new vector for insertion.
      llvm::SmallVector<int64_t, 8> vec(symInRow.begin(), symInRow.end());
      vec.push_back(row.back());

      if (isEq)
        cst.addEquality(vec);
      else
        cst.addInequality(vec);
    }
  }

  cst.removeRedundantConstraints();

  return cst;
}

/// Make sure that cstA and cstB has the same set of symbols, ordered in the
/// same sequence.
static void mergeAndAlignSymbols(FlatAffineConstraints &cstA,
                                 const FlatAffineConstraints &cstB) {
  llvm::SmallVector<mlir::Value, 8> symbols;
  cstB.getIdValues(cstB.getNumDimIds(), cstB.getNumDimAndSymbolIds(), &symbols);

  for (unsigned i = 0; i < symbols.size(); i++) {
    unsigned posA;

    mlir::Value valB = symbols[i];
    if (cstA.findId(valB, &posA)) {
      if (posA == i + cstA.getNumDimIds())
        continue;
      cstA.swapId(posA, i + cstA.getNumDimIds());
    } else {
      cstA.addSymbolId(i, valB);
    }
  }
}

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

static void setDomainAttrs(const OslScopStmtMap &sm,
                           const OslScopSymbolTable &st,
                           const FlatAffineConstraints &ctx) {
  for (const auto &key : sm.getKeys()) {
    const ScopStmt &s = sm.lookup(key);
    FlatAffineConstraints domain(s.getDomain());

    mergeAndAlignSymbols(domain, ctx);

    mlir::CallOp caller = s.getCaller();
    IntegerSet iset = domain.getAsIntegerSet(caller.getContext());

    caller.setAttr(OslScop::SCOP_STMT_DOMAIN_NAME, IntegerSetAttr::get(iset));

    llvm::SmallVector<mlir::Attribute, 8> symbolAttrs;
    for (unsigned i = 0; i < domain.getNumDimAndSymbolIds(); i++) {
      Optional<mlir::Value> id = domain.getId(i);
      if (id.hasValue()) {
        mlir::Value value = id.getValue();
        symbolAttrs.push_back(
            StringAttr::get(st.lookup(value), caller.getContext()));
      } else {
        symbolAttrs.push_back(StringAttr::get("", caller.getContext()));
      }
    }

    caller.setAttr(OslScop::SCOP_STMT_DOMAIN_SYMBOLS_NAME,
                   ArrayAttr::get(symbolAttrs, caller.getContext()));
  }
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
      mergeAndAlignSymbols(cst, ctx);

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

  // -------------------------- Setup OpenScop content ------------------------

  // Context relation.
  scop->addContextRelation(getOslContext(ctx));
  scop->addParameterNames(symbolTable);
  scop->addScatnames(scatTreeRoot);
  scop->addArrays(symbolTable);

  // Put the content of scopStmtMap into scop. The statements will be inserted
  // in the same order as they are iterated.
  for (const auto &key : scopStmtMap.getKeys()) {
    const ScopStmt &scopStmt = scopStmtMap.lookup(key);

    // We create a copy of the domain. This copy will be later updated.
    FlatAffineConstraints domain(scopStmt.getDomain());
    // According to the OpenScop specification: "The number of parameters should
    // be the same for all relations in the entire OpenScop file or data
    // structure." Therefore, we should merge and align all the parameters
    // (symbols) in domains with the context.
    mergeAndAlignSymbols(domain, ctx);

    llvm::SmallVector<unsigned, 8> scats;
    scatTreeRoot.getPathIds(scopStmt.getCaller(), scats);

    osl_statement *oslStmt = scop->createStatement();

    // Domain relation
    scop->addDomainRelation(oslStmt, domain);

    // Scattering relation
    scop->addScatteringRelation(oslStmt, domain, scats);

    // Access relations
    scopStmt.getCallee()->walk([&](Operation *op) {
      if (isa<mlir::AffineReadOpInterface, mlir::AffineWriteOpInterface>(op)) {
        Value memref;
        MemRefAccess access(op);
        FlatAffineConstraints cst;

        // Get the value-replaced (by the operand of the caller) access map and
        // memref. All the values in the vMap and memref can be found in the
        // symbol table.
        scopStmt.getAccessConstraints(access, symbolTable, cst, domain);

        // NOTE: The array identifier in OpenScop should start from 1.
        scop->addAccessRelation(
            /*stmt=*/oslStmt, /*isRead=*/isa<mlir::AffineReadOpInterface>(op),
            /*domain=*/domain, /*domain=*/cst);
      }
    });

    // Body extension.
    scop->addBody(oslStmt, scopStmt, symbolTable);
  }

  // TODO: add a flag to trigger these annotation functions.
  setSymbolAttrs(f, symbolTable, ctx);
  setDomainAttrs(scopStmtMap, symbolTable, ctx);
  setScatTreeAttrs(scopStmtMap, scatTreeRoot);
  setAccessAttrs(scopStmtMap, symbolTable, ctx);

  return scop;
}
