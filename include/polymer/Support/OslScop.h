//===- OslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "polymer/Support/ScatTree.h"
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

/// Container for the symbol table that maps from symbols used in OslScop to
/// MLIR values.
class OslScopSymbolTable {
public:
  using Container = llvm::StringMap<mlir::Value>;

  enum SymbolType { NOT_A_SYMBOL, MEMREF, INDVAR, PARAMETER, CONSTANT };

  /// Get a const reference to the symbol table.
  const Container &getSymbolTable() const;

  /// Find symbol from the symbol table for the given Value.  Return an empty
  /// symbol if not found.
  llvm::StringRef lookup(mlir::Value value) const;
  llvm::StringRef lookup(mlir::Value value, unsigned *numSymbolsOfType) const;

  /// Return the found symbol without its prefix.
  int64_t lookupId(mlir::Value) const;

  /// Find the symbol for the given Value. If not exists, create a new one based
  /// on the hardcoded rule.
  llvm::StringRef lookupOrCreate(mlir::Value value);

  /// Get the symbol prefix ('A', 'i', 'P', etc.) based on the symbol type.
  static llvm::StringRef getPrefix(SymbolType type);
  /// Get the symbol prefix ('A', 'i', 'P', etc.) based on the value type.
  static llvm::StringRef getPrefix(mlir::Value value);

  /// Get the symbol type based on the symbol content.
  static SymbolType getType(llvm::StringRef symbol);
  /// Get the symbol type based on the MLIR value.
  static SymbolType getType(mlir::Value value);

  /// Remove the prefix and leave only the numerical part.
  static llvm::StringRef dropPrefix(llvm::StringRef symbol, SymbolType type);

private:
  Container symbolTable;
};

/// Container for the mapping between symbols and ScopStmts.
class OslScopStmtMap {
public:
  using Container = llvm::StringMap<ScopStmt>;
  using Symbols = llvm::SmallVector<llvm::StringRef, 8>;

  using iterator = Container::iterator;
  using const_iterator = Container::const_iterator;

  /// Iterators over the internal map. Note that they do not keep the insertion
  /// order.
  iterator begin() { return map.begin(); }
  iterator end() { return map.end(); }
  const_iterator begin() const { return map.begin(); }
  const_iterator end() const { return map.end(); }

  /// Lookup by ScopStmt symbol.
  const ScopStmt &lookup(llvm::StringRef key) const;

  /// Get the size of the map.
  unsigned size() const { return map.size(); }

  /// Insert a ScopStmt into the map. A new entry will be initialized in
  /// scopStmtMap and its symbol will be appended to scopStmtSymbols.
  void insert(ScopStmt scopStmt);
  void insert(mlir::CallOp caller, mlir::FuncOp callee);

  /// Build context constraints from the scopStmtMap. The context is basically a
  /// union of all domain constraints. Its ID values will be used to derive the
  /// symbol table. Its constraints, however, can only be used once we project
  /// out all the dim values, i.e., just leave the constraints only on symbols.
  mlir::FlatAffineConstraints getContext() const;

  /// Return the keys as stored in scopStmtSymbols.
  const Symbols &getKeys() const;

private:
  Container map;
  Symbols keys;
};

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
  static constexpr const char *const SCOP_STMT_DOMAIN_NAME = "scop.domain";
  static constexpr const char *const SCOP_STMT_DOMAIN_SYMBOLS_NAME =
      "scop.domain_symbols";
  static constexpr const char *const SCOP_STMT_ACCESS_NAME = "scop.access";
  static constexpr const char *const SCOP_STMT_ACCESS_SYMBOLS_NAME =
      "scop.access_symbols";
  static constexpr const char *const SCOP_STMT_SCATS_NAME = "scop.scats";
  static constexpr const char *const SCOP_IV_NAME_ATTR_NAME = "scop.iv_name";
  static constexpr const char *const SCOP_PARAM_NAMES_ATTR_NAME =
      "scop.param_names";
  static constexpr const char *const SCOP_ARG_NAMES_ATTR_NAME =
      "scop.arg_names";

  enum SymbolType { NOT_A_SYMBOL, MEMREF, INDVAR, PARAMETER, CONSTANT };

private:
  /// The osl_scop object being managed.
  osl_scop_unique_ptr scop;

public:
  OslScop();
  OslScop(osl_scop *scop) : scop(osl_scop_unique_ptr{scop, osl_scop_free}) {}
  OslScop(osl_scop_unique_ptr scop) : scop(std::move(scop)) {}

  OslScop(const OslScop &) = delete;
  OslScop &operator=(const OslScop &) = delete;

  ~OslScop();

  /// Get the raw scop pointer.
  osl_scop *get() { return scop.get(); }

  /// Print the content of the Scop to the stdout. By default print to stderr.
  void print(FILE *fp = stderr) const;

  /// Validate whether the scop is well-formed. This will call the
  /// osl_scop_integrity_check function from OpenScop.
  bool validate() const;

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
                             llvm::ArrayRef<mlir::Operation *> ops,
                             mlir::Operation *caller);

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
};

} // namespace polymer

#endif
