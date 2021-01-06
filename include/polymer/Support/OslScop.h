//===- OslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "polymer/Support/ScatUtils.h"
#include "polymer/Support/ScopStmt.h"

#include "osl/osl.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringMap.h"

namespace mlir {
struct LogicalResult;
class FlatAffineConstraints;
class Value;
class Operation;
class AffineValueMap;
class CallOp;
class FuncOp;
} // namespace mlir

namespace polymer {

/// A wrapper for the osl_scop struct in the openscop library. It mainly
/// provides functionalities for accessing the contents in a osl_scop, and
/// the methods for adding new relations. It also holds a symbol table that maps
/// between a symbol in the OpenScop representation and a Value in the original
/// MLIR input. It manages the life-cycle of the osl_scop object passed in
/// through unique_ptr.
class OslScop {
public:
  using SymbolTable = llvm::StringMap<mlir::Value>;
  using ScopStmtMap = llvm::StringMap<ScopStmt>;
  using osl_scop_unique_ptr =
      std::unique_ptr<osl_scop_t, decltype(osl_scop_free) *>;

  static constexpr const char *const SCOP_STMT_ATTR_NAME = "scop.stmt";
  static constexpr const char *const SCOP_IV_NAME_ATTR_NAME = "scop.iv_name";
  static constexpr const char *const SCOP_PARAM_NAMES_ATTR_NAME =
      "scop.param_names";
  static constexpr const char *const SCOP_ARG_NAMES_ATTR_NAME =
      "scop.arg_names";

  enum SymbolType { NOT_A_SYMBOL, MEMREF, INDVAR, PARAMETER, CONSTANT };

private:
  /// The osl_scop object being managed.
  osl_scop_unique_ptr scop;
  /// The symbol table that maps from symbols in the OpenScop (e.g., scatnames,
  /// array), to the corresponding mlir::Value.
  SymbolTable symbolTable;
  /// Root to the scattering tree.
  std::unique_ptr<ScatTreeNode> scatTreeRoot;
  /// Mapping between ScopStmts and their symbols.
  ScopStmtMap scopStmtMap;
  /// Keep the ScopStmt symbols in their discovery order.
  llvm::SmallVector<llvm::StringRef, 8> scopStmtSymbols;

public:
  OslScop();
  OslScop(osl_scop *scop)
      : scop(osl_scop_unique_ptr{scop, osl_scop_free}),
        scatTreeRoot(std::make_unique<ScatTreeNode>()) {}
  OslScop(osl_scop_unique_ptr scop)
      : scop(std::move(scop)), scatTreeRoot(std::make_unique<ScatTreeNode>()) {}

  OslScop(const OslScop &) = delete;
  OslScop &operator=(const OslScop &) = delete;

  ~OslScop();

  /// Initialize the internal data structures. You should only call this once
  /// all ScopStmts are inserted into the scopStmtMap.
  void initialize();

  /// Get the raw scop pointer.
  osl_scop *get() { return scop.get(); }

  /// Print the content of the Scop to the stdout. By default print to stderr.
  void print(FILE *fp = stderr) const;

  /// Validate whether the scop is well-formed. This will call the
  /// osl_scop_integrity_check function from OpenScop.
  bool validate() const;

  /// ------------------------- ScopStmtMap ------------------------------------

  /// Return a const reference to the internal scopStmtMap data.
  const ScopStmtMap &getScopStmtMap() const;

  /// Add a ScopStmt into the map. A new entry will be initialized in
  /// scopStmtMap and its symbol will be appended to scopStmtSymbols.
  void addScopStmt(mlir::CallOp caller, mlir::FuncOp callee);

  /// Build context constraints from the scopStmtMap. The context is basically a
  /// union of all domain constraints. Its ID values will be used to derive the
  /// symbol table. Its constraints, however, can only be used once we project
  /// out all the dim values, i.e., just leave the constraints only on symbols.
  void getContextConstraints(mlir::FlatAffineConstraints &ctx) const;

  /// ------------------------- Statements -------------------------------------

  /// Simply create a new statement in the linked list scop->statement.
  osl_statement *createStatement();

  /// Get statement by index.
  osl_statement *getStatement(unsigned index) const;

  /// Get the total number of statements
  unsigned getNumStatements() const;

  /// ------------------------- Relations -------------------------------------

  /// TODO: Should remove this.
  void addRelation(int target, int type, int numRows, int numCols,
                   int numOutputDims, int numInputDims, int numLocalDims,
                   int numParams, llvm::ArrayRef<int64_t> eqs,
                   llvm::ArrayRef<int64_t> inEqs);

  /// Add the relation defined by the context constraints (cst) to the context
  /// of the current scop. The cst passed in should contain all the parameters
  /// in all the domain relations. Also, the order of parameters in the cst
  /// should be consistent with all the domain constraints. There shouldn't be
  /// any dim or local IDs in the constraint, only symbol IDs (parameters) are
  /// allowed.
  void addContextRelation(const mlir::FlatAffineConstraints &cst);

  /// Add the domain relation to the statement denoted by ID. We don't do any
  /// additional validation in this function. We simply get the flattened array
  /// of equalities and inequalities from the cst and add it to the target
  /// statement. stmtId starts from 0.
  void addDomainRelation(int stmtId, const mlir::FlatAffineConstraints &cst);

  /// Add the scattering relation to the target statement (given by stmtId). cst
  /// is the domain of the statement that this scattering relation is added to.
  /// ops are the enclosing affine.for of the current statement.
  void addScatteringRelation(int stmtId, const mlir::FlatAffineConstraints &cst,
                             llvm::ArrayRef<mlir::Operation *> ops);

  /// Add the access relation to the target statement (given by stmtId).
  void addAccessRelation(int stmtId, bool isRead, mlir::Value memref,
                         mlir::AffineValueMap &vMap,
                         mlir::FlatAffineConstraints &cst);

  /// ------------------------- Extensions ------------------------------------

  /// TODO: Should remove this soon.
  void addGeneric(int target, llvm::StringRef tag, llvm::StringRef content);

  /// Add an extension generic content to the whole osl_scop. tag specifies what
  /// the generic type is.
  void addExtensionGeneric(llvm::StringRef tag, llvm::StringRef content);

  /// Add a  generic content to the beginning of the whole osl_scop. tag
  /// specifies the type of the extension. The content has a space separated
  /// list of parameter names.
  void addParametersGeneric(llvm::StringRef tag, llvm::StringRef content);

  /// Add a generic to a single statement, which can be <body>, for example. tag
  /// specifies what the generic type is.
  void addStatementGeneric(int stmtId, llvm::StringRef tag,
                           llvm::StringRef content);

  /// Check whether the name refers to a symbol.
  bool isSymbol(llvm::StringRef name);

  /// Get extension by the tag name. tag can be strings like "body", "array",
  /// etc. This function goes through the whole scop to find is there an
  /// extension that matches the tag.
  osl_generic *getExtension(llvm::StringRef tag) const;

  /// Add parameter names to the <strings> extension of scop->parameters from
  /// the symbol table.
  void addParameterNamesFromSymbolTable();

  /// Create strings based on the depth of the scat tree and add them
  /// to the <scatnames> extension.
  void addScatnamesExtensionFromScatTree();

  /// Create the <arrays> extension from the symbol table.
  void addArraysExtensionFromSymbolTable();

  /// Add the <body> extension content from the given ScopStmt object.
  void addBodyExtension(int stmtId, const ScopStmt &stmt);

  /// ------------------------- Symbol Table ----------------------------------

  /// Get a const reference to the symbol table.
  const SymbolTable &getSymbolTable() const;

  /// Find symbol from the symbol table for the given Value.  Return an empty
  /// symbol if not found.
  llvm::StringRef getSymbol(mlir::Value value) const;
  llvm::StringRef getSymbol(mlir::Value value,
                            unsigned *numSymbolsOfType) const;

  /// Find the symbol for the given Value. If not exists, create a new one based
  /// on the hardcoded rule.
  llvm::StringRef getOrCreateSymbol(mlir::Value value);

  /// Get the symbol prefix ('A', 'i', 'P', etc.) based on the symbol type.
  llvm::StringRef getSymbolPrefix(SymbolType type) const;
  /// Get the symbol prefix ('A', 'i', 'P', etc.) based on the value type.
  llvm::StringRef getSymbolPrefix(mlir::Value value) const;

  /// Get the symbol type based on the symbol content.
  SymbolType getSymbolType(llvm::StringRef symbol) const;
  /// Get the symbol type based on the MLIR value.
  SymbolType getSymbolType(mlir::Value value) const;

  /// Initialize symbol table. Parameters and induction variables can be found
  /// from the context (ctx) derived from scopStmtMap.
  void initSymbolTable(const mlir::FlatAffineConstraints &ctx);
};

} // namespace polymer

#endif
