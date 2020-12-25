//===- ScatUtils.cc ---------------------------------------------*- C++ -*-===//
//
// This file implements the C++ wrapper for the Scop scattering.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/ScatUtils.h"

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
  ScatTreeNodeImpl() {}
  ScatTreeNodeImpl(mlir::Value iv) : iv(iv) {}

  /// Get the tree depth starting from the current node.
  unsigned getDepth() const;

  /// Insert a path starting from the current node. The path is represented by a
  /// list of induction variables. scatIds record as which child each indvar is
  /// inserted.
  void insertPath(llvm::ArrayRef<mlir::Value> ivs,
                  llvm::SmallVectorImpl<unsigned> &scatIds);

  /// If iv is NULL.
  bool isRootOrLeaf() const;

  /// Return the ID of the child by its induction variable. If not found, return
  /// nullptr.
  Optional<unsigned> findChildId(mlir::Value) const;

  /// Add child. If iv is not provided, this new child is a leaf node.
  unsigned addChild(mlir::Value iv = nullptr);

  /// Children of the current node.
  llvm::SmallVector<std::unique_ptr<ScatTreeNodeImpl>, 8> children;
  /// Mapping from IV to child ID (DEPRECATED).
  llvm::DenseMap<mlir::Value, unsigned> valueIdMap;
  /// Induction variable stored.
  mlir::Value iv;
};

bool ScatTreeNodeImpl::isRootOrLeaf() const { return iv == nullptr; }

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

unsigned ScatTreeNodeImpl::addChild(mlir::Value iv) {
  assert(!findChildId(iv) &&
         "iv should be a nullptr, or it has not been inserted before.");

  std::unique_ptr<ScatTreeNodeImpl> child =
      std::make_unique<ScatTreeNodeImpl>(iv);
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
                                  llvm::SmallVectorImpl<unsigned> &scatIds) {
  // We cannot distinguish between a leaf and a root node that has no children,
  // so here we use a loose condition.
  assert(isRootOrLeaf() && "Should only insert path after a root/leaf node.");

  scatIds.clear();
  scatIds.resize(ivs.size() + 1);

  ScatTreeNodeImpl *curr = this;
  for (unsigned i = 0; i < ivs.size(); i++) {
    mlir::Value iv = ivs[i];
    Optional<unsigned> id = curr->findChildId(iv);
    if (!id)
      id = curr->addChild(iv);

    curr = curr->children[id.getValue()].get();
    scatIds[i] = id.getValue();
  }

  // Add the final leaf node.
  scatIds[ivs.size()] = curr->addChild();
}

/// Insert a statement characterized by its enclosing operations into a
/// "scattering tree". This is done by iterating through every enclosing for-op
/// from the outermost to the innermost, and we try to traverse the tree by the
/// IVs of these ops. If an IV does not exist, we will insert it into the tree.
/// After that, we insert the current load/store statement into the tree as a
/// leaf. In this progress, we keep track of all the IDs of each child we meet
/// and the final leaf node, which will be used as the scattering.
static void insertStatement(ScatTreeNodeImpl *root,
                            ArrayRef<Operation *> enclosingOps,
                            SmallVectorImpl<unsigned> &scats) {
  ScatTreeNodeImpl *curr = root;

  for (unsigned i = 0, e = enclosingOps.size(); i < e; i++) {
    Operation *op = enclosingOps[i];
    // We only handle for op here.
    // TODO: is it necessary to deal with if?
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      SmallVector<mlir::Value, 4> indices;
      extractForInductionVars(forOp, &indices);

      for (const auto &iv : indices) {
        auto it = curr->valueIdMap.find(iv);
        if (it != curr->valueIdMap.end()) {
          // Add a new element to the scattering.
          scats.push_back(it->second);
          // move to the next IV along the tree.
          curr = curr->children[it->second].get();
        } else {
          // No existing node for such IV is found, create a new one.
          auto node = std::make_unique<ScatTreeNodeImpl>(iv);

          // Then insert the newly created node into the children set, update
          // the value to child ID map, and move the cursor to this new node.
          curr->children.push_back(std::move(node));
          unsigned valueId = curr->children.size() - 1;
          curr->valueIdMap[iv] = valueId;
          scats.push_back(valueId);
          curr = curr->children.back().get();
        }
      }
    }
  }

  // Append the leaf node for statement
  auto leaf = std::make_unique<ScatTreeNodeImpl>();
  curr->children.push_back(std::move(leaf));
  scats.push_back(curr->children.size() - 1);
}

ScatTreeNode::ScatTreeNode() : impl{std::make_unique<ScatTreeNodeImpl>()} {}
ScatTreeNode::ScatTreeNode(mlir::Value value)
    : impl{std::make_unique<ScatTreeNodeImpl>(value)} {}
ScatTreeNode::~ScatTreeNode() = default;
ScatTreeNode::ScatTreeNode(ScatTreeNode &&) = default;
ScatTreeNode &ScatTreeNode::operator=(ScatTreeNode &&) = default;

void ScatTreeNode::insertScopStmt(llvm::ArrayRef<mlir::Operation *> ops,
                                  llvm::SmallVectorImpl<unsigned> &scats) {
  insertStatement(impl.get(), ops, scats);
}

void ScatTreeNode::insertOperation(mlir::Operation *op,
                                   llvm::SmallVectorImpl<unsigned> &scats) {
  llvm::SmallVector<mlir::Value, 4> ivs;
  llvm::SmallVector<mlir::Operation *, 8> enclosingOps;

  getEnclosingAffineForAndIfOps(*op, &enclosingOps);
  for (mlir::Operation *op : enclosingOps)
    if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(op))
      ivs.push_back(forOp.getInductionVar());

  impl->insertPath(ivs, scats);
}

unsigned ScatTreeNode::getDepth() const { return impl->getDepth(); }

} // namespace polymer
