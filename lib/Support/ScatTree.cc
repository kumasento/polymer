//===- ScatUtils.cc ---------------------------------------------*- C++ -*-===//
//
// This file implements the C++ wrapper for the Scop scattering.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/ScatTree.h"

#include <memory>

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace polymer;
using namespace mlir;
using namespace llvm;

namespace polymer {

/// The internal implementation of ScatTreeNode.
class ScatTreeNodeImpl {
public:
  /// Root node.
  ScatTreeNodeImpl() : iv(nullptr), op(nullptr) {}
  /// Leaf node
  ScatTreeNodeImpl(mlir::Operation *op) : iv(nullptr), op(op) {}
  /// Other nodes in the tree.
  ScatTreeNodeImpl(mlir::Value iv) : iv(iv), op(nullptr) {}

  /// Get the tree depth starting from the current node.
  unsigned getDepth() const;

  /// Insert a path starting from the current node. The path is represented by a
  /// list of induction variables. scatIds record as which child each indvar is
  /// inserted.
  void insertPath(llvm::ArrayRef<mlir::Value> ivs, mlir::Operation *op);
  void insertPath(llvm::ArrayRef<mlir::Value> ivs, mlir::Operation *op,
                  llvm::SmallVectorImpl<unsigned> &scatIds);

  /// Get the IDs along the path starting from the tree root, ending at the
  /// given statement op.
  void getPathIds(llvm::ArrayRef<mlir::Value> ivs, mlir::Operation *op,
                  llvm::SmallVectorImpl<unsigned> &scatIds) const;

  /// If iv is NULL and op is not NULL.
  bool isLeaf() const;
  /// If iv is NULL and op is NULL.
  bool isRoot() const;
  /// isLeaf() || isRoot()
  bool isRootOrLeaf() const;

  /// Return the ID of the child by its induction variable. If not found, return
  /// nullptr.
  Optional<unsigned> findChildId(mlir::Value) const;
  Optional<unsigned> findChildId(mlir::Operation *) const;

  /// Add a non-leaf child, returns its ID in its parent children.
  unsigned addChild(mlir::Value iv);
  /// Add a leaf child, returns its ID in its parent children.
  unsigned addChild(mlir::Operation *op);

private:
  /// Children of the current node.
  llvm::SmallVector<std::unique_ptr<ScatTreeNodeImpl>, 8> children;

  /// Induction variable stored.
  const mlir::Value iv;
  /// The statement operation (CallerOp) stored.
  const mlir::Operation *op;
};

bool ScatTreeNodeImpl::isLeaf() const { return iv == nullptr && op != nullptr; }
bool ScatTreeNodeImpl::isRoot() const { return iv == nullptr && op == nullptr; }
bool ScatTreeNodeImpl::isRootOrLeaf() const { return isLeaf() || isRoot(); }

Optional<unsigned> ScatTreeNodeImpl::findChildId(mlir::Value iv) const {
  Optional<unsigned> id;
  if (!iv)
    return id;

  for (unsigned i = 0; i < children.size(); i++)
    if (children[i]->iv == iv) {
      id = i;
      break;
    }

  return id;
}

Optional<unsigned> ScatTreeNodeImpl::findChildId(mlir::Operation *op) const {
  Optional<unsigned> id;
  if (!op)
    return id;

  for (unsigned i = 0; i < children.size(); i++)
    if (children[i]->op == op) {
      id = i;
      break;
    }

  return id;
}

unsigned ScatTreeNodeImpl::addChild(mlir::Value iv) {
  assert(!findChildId(iv) &&
         "iv should be a nullptr, or it has not been inserted before.");

  std::unique_ptr<ScatTreeNodeImpl> child =
      std::make_unique<ScatTreeNodeImpl>(iv);
  children.push_back(std::move(child));
  return children.size() - 1;
}

unsigned ScatTreeNodeImpl::addChild(mlir::Operation *op) {
  assert(!findChildId(op) &&
         "op should be a nullptr, or it has not been inserted before.");

  std::unique_ptr<ScatTreeNodeImpl> child =
      std::make_unique<ScatTreeNodeImpl>(op);
  children.push_back(std::move(child));
  return children.size() - 1;
}

unsigned ScatTreeNodeImpl::getDepth() const {
  // Each NodeDepthPair stores the pointer to the node and the depth of that
  // node. We use the nodeDepthPairs as a stack to implement the Depth-First
  // Search based algorithm without using recursion.
  using NodeDepthPair = std::pair<const ScatTreeNodeImpl *, unsigned>;
  llvm::SmallVector<NodeDepthPair, 8> nodeDepthPairs;

  // We search for every node in the tree following the DFS order, and keep the
  // largest depth value during searching.
  nodeDepthPairs.push_back(std::make_pair(this, 1));
  unsigned maxDepth = 1;

  while (!nodeDepthPairs.empty()) {
    NodeDepthPair curr = nodeDepthPairs.pop_back_val();
    maxDepth = std::max(maxDepth, curr.second);

    for (const std::unique_ptr<ScatTreeNodeImpl> &child : curr.first->children)
      nodeDepthPairs.push_back(std::make_pair(child.get(), curr.second + 1));
  }

  return maxDepth;
}

void ScatTreeNodeImpl::insertPath(llvm::ArrayRef<mlir::Value> ivs,
                                  mlir::Operation *op) {
  assert(isRoot() && "Should only insert path after a leaf node.");

  ScatTreeNodeImpl *curr = this;
  for (unsigned i = 0; i < ivs.size(); i++) {
    mlir::Value iv = ivs[i];
    Optional<unsigned> id = curr->findChildId(iv);
    if (!id)
      id = curr->addChild(iv);

    curr = curr->children[id.getValue()].get();
  }

  // Add the final leaf node.
  curr->addChild(op);
}

void ScatTreeNodeImpl::insertPath(llvm::ArrayRef<mlir::Value> ivs,
                                  mlir::Operation *op,
                                  llvm::SmallVectorImpl<unsigned> &scatIds) {
  insertPath(ivs, op);
  getPathIds(ivs, op, scatIds);
}

void ScatTreeNodeImpl::getPathIds(
    llvm::ArrayRef<mlir::Value> ivs, mlir::Operation *op,
    llvm::SmallVectorImpl<unsigned> &scatIds) const {
  scatIds.clear();
  scatIds.resize(ivs.size() + 1);

  const ScatTreeNodeImpl *curr = this;
  for (unsigned i = 0; i < ivs.size(); i++) {
    mlir::Value iv = ivs[i];
    // This line would trigger error if the given IV is not found.
    unsigned id = curr->findChildId(iv).getValue();

    scatIds[i] = id;
    curr = curr->children[id].get();
  }

  scatIds[ivs.size()] = curr->findChildId(op).getValue();
}

/// ------------------------------ ScatTreeNode -------------------------------

ScatTreeNode::ScatTreeNode() : impl{std::make_unique<ScatTreeNodeImpl>()} {}
ScatTreeNode::~ScatTreeNode() = default;
ScatTreeNode::ScatTreeNode(ScatTreeNode &&) = default;
ScatTreeNode &ScatTreeNode::operator=(ScatTreeNode &&) = default;

static void getEnclosingIndvars(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                                llvm::SmallVectorImpl<mlir::Value> &ivs) {
  for (mlir::Operation *op : enclosingOps)
    if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(op))
      ivs.push_back(forOp.getInductionVar());
}

static void getEnclosingIndvars(mlir::Operation *op,
                                llvm::SmallVectorImpl<mlir::Value> &ivs) {
  llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
  getEnclosingAffineForAndIfOps(*op, &enclosingOps);
  getEnclosingIndvars(enclosingOps, ivs);
}

void ScatTreeNode::insertPath(mlir::Operation *op) {
  llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
  getEnclosingAffineForAndIfOps(*op, &enclosingOps);

  insertPath(enclosingOps, op);
}

void ScatTreeNode::insertPath(mlir::Operation *op,
                              llvm::SmallVectorImpl<unsigned> &scats) {
  insertPath(op);
  getPathIds(op, scats);
}

void ScatTreeNode::insertPath(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                              mlir::Operation *op) {
  llvm::SmallVector<mlir::Value, 4> ivs;
  getEnclosingIndvars(enclosingOps, ivs);

  impl->insertPath(ivs, op);
}

void ScatTreeNode::insertPath(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                              mlir::Operation *op,
                              llvm::SmallVectorImpl<unsigned> &scats) {
  insertPath(enclosingOps, op);
  getPathIds(enclosingOps, op, scats);
}

void ScatTreeNode::getPathIds(mlir::Operation *op,
                              llvm::SmallVectorImpl<unsigned> &scats) const {
  llvm::SmallVector<mlir::Value, 8> ivs;
  getEnclosingIndvars(op, ivs);

  impl->getPathIds(ivs, op, scats);
}

void ScatTreeNode::getPathIds(llvm::ArrayRef<mlir::Operation *> enclosingOps,
                              mlir::Operation *op,
                              llvm::SmallVectorImpl<unsigned> &scats) const {
  llvm::SmallVector<mlir::Value, 8> ivs;
  getEnclosingIndvars(enclosingOps, ivs);

  impl->getPathIds(ivs, op, scats);
}

unsigned ScatTreeNode::getDepth() const { return impl->getDepth(); }

} // namespace polymer
