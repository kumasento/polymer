//===- OslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "polymer/Support/ScatUtils.h"

#include "mlir/Support/LLVM.h"
#include "osl/osl.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
struct LogicalResult;
class FlatAffineConstraints;
class Operation;
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
  using osl_scop_unique_ptr =
      std::unique_ptr<osl_scop_t, decltype(osl_scop_free) *>;

  OslScop();
  OslScop(osl_scop *scop)
      : scop(osl_scop_unique_ptr{scop, osl_scop_free}),
        scatTreeRoot(std::make_unique<ScatTreeNode>()) {}
  OslScop(osl_scop_unique_ptr scop)
      : scop(std::move(scop)), scatTreeRoot(std::make_unique<ScatTreeNode>()) {}

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

  /// Simply create a new statement in the linked list scop->statement.
  void createStatement();
  /// Get statement by index.
  osl_statement *getStatement(unsigned index) const;
  /// Get the total number of statements
  unsigned getNumStatements() const;

  /// Create a new relation and initialize its contents. The new relation will
  /// be created under the scop member.
  /// The target here is an index:
  /// 1) if it's 0, then it means the context;
  /// 2) otherwise, if it is a positive number, it corresponds to a statement of
  /// id=(target-1).
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

  /// Add a new generic field to a statement. `target` gives the statement ID.
  /// `content` specifies the data field in the generic.
  void addGeneric(int target, llvm::StringRef tag, llvm::StringRef content);

  /// Check whether the name refers to a symbol.
  bool isSymbol(llvm::StringRef name);

  /// Get extension by interface name
  osl_generic *getExtension(llvm::StringRef interface) const;

private:
  /// The osl_scop object being managed.
  osl_scop_unique_ptr scop;
  /// Root to the scattering tree.
  std::unique_ptr<ScatTreeNode> scatTreeRoot;
};

} // namespace polymer

#endif
