//===- EmitOpenScop.cc ------------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for emitting OpenScop representation from
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScop.h"
#include "polymer/Support/OslScopBuilder.h"
#include "polymer/Target/OpenScop.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "osl/osl.h"

#include <memory>

using namespace mlir;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "emit-openscop"

namespace {

/// This class maintains the state of a working emitter.
class OpenScopEmitterState {
public:
  explicit OpenScopEmitterState(raw_ostream &os) : os(os) {}

  /// The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIdent = 0; // TODO: may not need this.

private:
  OpenScopEmitterState(const OpenScopEmitterState &) = delete;
  void operator=(const OpenScopEmitterState &) = delete;
};

/// Base class for various OpenScop emitters.
class OpenScopEmitterBase {
public:
  explicit OpenScopEmitterBase(OpenScopEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitOpError(message);
  }

  /// All of the mutable state we are maintaining.
  OpenScopEmitterState &state;

  /// The stream to emit to.
  raw_ostream &os;

private:
  OpenScopEmitterBase(const OpenScopEmitterBase &) = delete;
  void operator=(const OpenScopEmitterBase &) = delete;
};

/// Emit OpenScop representation from an MLIR module.
class ModuleEmitter : public OpenScopEmitterBase {
public:
  explicit ModuleEmitter(OpenScopEmitterState &state)
      : OpenScopEmitterBase(state) {}

  /// Emit OpenScop definitions for all functions in the given module.
  void emitMLIRModule(ModuleOp module,
                      llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops);

private:
  /// Emit a OpenScop definition for a single function.
  LogicalResult
  emitFuncOp(FuncOp func,
             llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops);
};

LogicalResult ModuleEmitter::emitFuncOp(
    mlir::FuncOp func, llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
  auto scop = createOpenScopFromFuncOp(func);
  if (!scop)
    return failure();

  scops.push_back(std::move(scop));
  return success();
}

/// The entry function to the current OpenScop emitter.
void ModuleEmitter::emitMLIRModule(
    ModuleOp module, llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops) {
  // Emit a single OpenScop definition for each function.
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<mlir::FuncOp>(op)) {
      if (failed(emitFuncOp(func, scops))) {
        state.encounteredError = true;
        return;
      }
    }
  }
}
} // namespace

std::unique_ptr<OslScop>
polymer::createOpenScopFromFuncOp(mlir::FuncOp funcOp) {
  return OslScopBuilder().build(funcOp);
}

/// TODO: should decouple emitter and openscop builder.
mlir::LogicalResult polymer::translateModuleToOpenScop(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
    llvm::raw_ostream &os) {
  OpenScopEmitterState state(os);
  ModuleEmitter(state).emitMLIRModule(module, scops);

  return success();
}

static LogicalResult emitOpenScop(ModuleOp module, llvm::raw_ostream &os) {
  llvm::SmallVector<std::unique_ptr<OslScop>, 8> scops;

  if (failed(translateModuleToOpenScop(module, scops, os)))
    return failure();

  for (auto &scop : scops)
    scop->print(stdout);

  return success();
}

void polymer::registerToOpenScopTranslation() {
  static TranslateFromMLIRRegistration toOpenScop("export-scop", emitOpenScop);
}
