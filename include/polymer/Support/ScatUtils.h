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
  ScatTreeNode();
  ScatTreeNode(mlir::Value iv);

  ~ScatTreeNode();

  ScatTreeNode(ScatTreeNode &&);
  ScatTreeNode(const ScatTreeNode &) = delete;
  ScatTreeNode &operator=(ScatTreeNode &&);
  ScatTreeNode &operator=(const ScatTreeNode &) = delete;

  /// Create a new leaf node. The insertion point is traced by the path starting
  /// from the current node, which is usually the root node, and followed by the
  /// IVs of the given enclosing affine.for operations (ops). If any IV is
  /// missing in the tree, we will insert a new one. The exact position of the
  /// IV at each level will be inserted into the scats array.
  void insertScopStmt(llvm::ArrayRef<mlir::Operation *> ops,
                      llvm::SmallVectorImpl<unsigned> &scats);

  /// Get the depth of the tree starting from this node.
  unsigned getDepth() const;

  /// Insert one operation into the scat tree based on its enclosing affine.for.
  void insertPath(mlir::Operation *op, llvm::SmallVectorImpl<unsigned> &scats);

  /// Insert a path of nodes into the scat tree based on the provided enclosing
  /// operations. We first extract all the induction variables from the
  /// affine.for provided in the given list of operations, then call the
  /// insertPath method.
  void insertPath(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                  llvm::SmallVectorImpl<unsigned> &scats);

private:
  std::unique_ptr<ScatTreeNodeImpl> impl;
};

} // namespace polymer

#endif
