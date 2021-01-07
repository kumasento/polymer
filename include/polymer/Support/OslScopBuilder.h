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
} // namespace mlir

namespace polymer {
class OslScop;

/// Build OslScop from FuncOp.
class OslScopBuilder {
public:
  /// Build a scop from a common FuncOp.
  std::unique_ptr<OslScop> build(mlir::FuncOp f);
};

} // namespace polymer

#endif
