//===- InvariantScopTransform.h - Invariant transform to OpenScop ---------===//
//
// This file declares the transformation between MLIR and OpenScop.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TRANSFORMS_INVARIANTSCOPTRANSFORM_H
#define POLYMER_TRANSFORMS_INVARIANTSCOPTRANSFORM_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace polymer {

void registerInvariantScopTransformPass();

} // namespace polymer

#endif
