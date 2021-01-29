//===- ScatUtils.h ----------------------------------------------*- C++ -*-===//
//
// This file declares the C++ wrapper for the Scop scattering.
//
//===----------------------------------------------------------------------===//
#ifndef POLYMER_SUPPORT_SCATUTILS_H
#define POLYMER_SUPPORT_SCATUTILS_H

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Value;
class Operation;
} // namespace mlir

namespace polymer {

class ScatTreeNodeImpl;

/// Tree that holds scattering information. This node can represent an induction
/// variable or a statement. A statement is ALWAYS constructed as a leaf node.
class ScatTreeNode {
public:
  /// Used to create a root node.
  ScatTreeNode();
  ~ScatTreeNode();

  ScatTreeNode(ScatTreeNode &&);
  ScatTreeNode(const ScatTreeNode &) = delete;
  ScatTreeNode &operator=(ScatTreeNode &&);
  ScatTreeNode &operator=(const ScatTreeNode &) = delete;

  /// Get the depth of the tree starting from this node.
  unsigned getDepth() const;

  /// Insert one operation into the scat tree based on its enclosing affine.for.
  void insertPath(mlir::Operation *op);
  void insertPath(mlir::Operation *op, llvm::SmallVectorImpl<unsigned> &scats);

  /// Insert a path of nodes into the scat tree based on the provided enclosing
  /// operations. We first extract all the induction variables from the
  /// affine.for provided in the given list of operations, then call the
  /// insertPath method.
  void insertPath(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                  mlir::Operation *op);
  void insertPath(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                  mlir::Operation *op, llvm::SmallVectorImpl<unsigned> &scats);

  /// Iterate from the tree root to collect all the child IDs along the path.
  void getPathIds(mlir::Operation *op,
                  llvm::SmallVectorImpl<unsigned> &scats) const;
  void getPathIds(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                  mlir::Operation *op,
                  llvm::SmallVectorImpl<unsigned> &scats) const;

private:
  std::unique_ptr<ScatTreeNodeImpl> impl;
};

} // namespace polymer

#endif
