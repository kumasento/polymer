#include "polymer/EmitOpenSCoP.h"

#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;
using namespace polymer;

namespace {

/// This class maintains the state of a working emitter.
class OpenSCoPEmitterState {
public:
  explicit OpenSCoPEmitterState(raw_ostream &os) : os(os) {}

  /// The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIdent = 0; // TODO: may not need this.

private:
  OpenSCoPEmitterState(const OpenSCoPEmitterState &) = delete;
  void operator=(const OpenSCoPEmitterState &) = delete;
};

/// Base class for various OpenSCoP emitters.
class OpenSCoPEmitterBase {
public:
  explicit OpenSCoPEmitterBase(OpenSCoPEmitterState &state)
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
  OpenSCoPEmitterState &state;

  /// The stream to emit to.
  raw_ostream &os;

private:
  OpenSCoPEmitterBase(const OpenSCoPEmitterBase &) = delete;
  void operator=(const OpenSCoPEmitterBase &) = delete;
};

/// Emit OpenSCoP representation from an MLIR module.
class ModuleEmitter : public OpenSCoPEmitterBase {
public:
  explicit ModuleEmitter(OpenSCoPEmitterState &state)
      : OpenSCoPEmitterBase(state) {}

  void emitMLIRModule(ModuleOp module);
};

/// The entry function to the current OpenSCoP emitter.
void ModuleEmitter::emitMLIRModule(ModuleOp module) {
  os << "<OpenSCoP>\n";

  os << "</OpenSCoP>\n";
}

} // namespace

LogicalResult polymer::emitOpenSCoP(ModuleOp module, llvm::raw_ostream &os) {
  OpenSCoPEmitterState state(os);
  ModuleEmitter(state).emitMLIRModule(module);

  return failure(state.encounteredError);
}

void polymer::registerOpenSCoPEmitterTranslation() {
  static TranslateFromMLIRRegistration toOpenSCoP("emit-openscop",
                                                  polymer::emitOpenSCoP);
}