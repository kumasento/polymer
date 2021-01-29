//===- OpenScop.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TARGET_OPENSCOP_H
#define POLYMER_TARGET_OPENSCOP_H

#include <memory>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class OwningModuleRef;
class MLIRContext;
class ModuleOp;
class FuncOp;
struct LogicalResult;
class Operation;
class Value;
} // namespace mlir

namespace polymer {

class OslScop;
class OslScopSymbolTable;

/// Create a single OpenScop (OslScop) from the given function. If nullptr is
/// returned, it means the given function is not of Scop.
std::unique_ptr<OslScop> createOpenScopFromFuncOp(mlir::FuncOp f);

/// Create a function (FuncOp) from the given OpenScop object in the given
/// module (ModuleOp).
mlir::Operation *createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop,
                                          OslScopSymbolTable &st,
                                          mlir::ModuleOp module,
                                          mlir::MLIRContext *context);
/// If we get OpenScop from a file, that is, no OslScopSymbolTable is available,
/// we use the following API which creates a dummy one (that could be updated
/// during the codegen).
mlir::Operation *createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop,
                                          mlir::ModuleOp module,
                                          mlir::MLIRContext *context);

mlir::OwningModuleRef translateOpenScopToModule(std::unique_ptr<OslScop> scop,
                                                mlir::MLIRContext *context);

mlir::LogicalResult translateModuleToOpenScop(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
    llvm::raw_ostream &os);

void registerToOpenScopTranslation();
void registerFromOpenScopTranslation();

} // namespace polymer

#endif
