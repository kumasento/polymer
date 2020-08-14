#ifndef POLYMER_EMITOPENSCOP_H
#define POLYMER_EMITOPENSCOP_H

#include "mlir/Support/LLVM.h"

namespace mlir {
struct LogicalResult;
class ModuleOp;
}  // namespace mlir

namespace polymer {

mlir::LogicalResult emitOpenSCoP(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerOpenSCoPEmitterTranslation();

}  // namespace polymer

#endif