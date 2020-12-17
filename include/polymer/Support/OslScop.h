//===- OslScop.h ------------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_OSLSCOP_H
#define POLYMER_SUPPORT_OSLSCOP_H

#include "mlir/Support/LLVM.h"
#include "osl/osl.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
struct LogicalResult;
}

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
      : scop(std::move(osl_scop_unique_ptr{scop, osl_scop_free})) {}
  OslScop(osl_scop_unique_ptr scop) : scop(std::move(scop)) {}

  OslScop(const OslScop &) = delete;
  OslScop &operator=(const OslScop &) = delete;

  ~OslScop();

  /// Get the raw scop pointer.
  osl_scop *get() { return scop.get(); }

  /// Print the content of the Scop to the stdout.
  void print();

  /// Validate whether the scop is well-formed.
  bool validate();

  /// Simply create a new statement in the linked list scop->statement.
  void createStatement();

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

  /// Add a new generic field to a statement. `target` gives the statement ID.
  /// `content` specifies the data field in the generic.
  void addGeneric(int target, llvm::StringRef tag, llvm::StringRef content);

  /// Check whether the name refers to a symbol.
  bool isSymbol(llvm::StringRef name);

  /// Get statement by index.
  mlir::LogicalResult getStatement(unsigned index, osl_statement **stmt);

  /// Get the total number of statements
  unsigned getNumStatements() const;

  /// Get extension by interface name
  osl_generic *getExtension(llvm::StringRef interface) const;

private:
  osl_scop_unique_ptr scop;
};

} // namespace polymer

#endif
