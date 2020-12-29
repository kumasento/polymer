//===- OslScopBuilder.h -----------------------------------------*- C++ -*-===//
//
// This file declares the class OslScopBuilder that builds OslScop from FuncOp.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_OSLSCOPBUILDER_H
#define POLYMER_SUPPORT_OSLSCOPBUILDER_H

#include <memory>

namespace mlir {
class FuncOp;
}

namespace polymer {

class OslScop;

/// Build OslScop from FuncOp.
class OslScopBuilder {
public:
  /// Build a scop from a common FuncOp.
  std::unique_ptr<OslScop> build(mlir::FuncOp f);

private:
  /// Find all statements that calls a scop.stmt.
  // void buildScopStmtMap(mlir::FuncOp f, OslScop::ScopStmtNames
  // *scopStmtNames,
  //                       OslScop::ScopStmtMap *scopStmtMap) const;

  /// Build the scop context. The domain of each scop stmt will be updated, by
  /// merging and aligning its IDs with the context as well.
  // void buildScopContext(OslScop *scop, OslScop::ScopStmtMap *scopStmtMap,
  //                       FlatAffineConstraints &ctx) const;
};

} // namespace polymer

#endif
