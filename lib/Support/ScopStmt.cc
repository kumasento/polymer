//===- ScopStmt.cc ----------------------------------------------*- C++ -*-===//
//
// This file declares the class ScopStmt.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/ScopStmt.h"
#include "polymer/Support/OslScop.h"
#include "polymer/Support/ScatTree.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "osl-scop"

using namespace llvm;
using namespace mlir;
using namespace polymer;

namespace polymer {

/// Internal implementation of the ScopStmt class.
class ScopStmtImpl {
public:
  using EnclosingOpList = SmallVector<Operation *, 8>;

  ScopStmtImpl(llvm::StringRef name, mlir::CallOp caller, mlir::FuncOp callee)
      : name(name), caller(caller), callee(callee) {
    // Initialize the domain constraints around the caller. The enclosing ops
    // will be figured out as well in this process.
    initializeDomainAndEnclosingOps();
  }

  /// Factory method that is consistent with the ScopStmt API.
  static std::unique_ptr<ScopStmtImpl> get(mlir::Operation *callerOp,
                                           mlir::Operation *calleeOp);

  /// A helper function that builds the domain constraints of the
  /// caller, and find and insert all enclosing for/if ops to enclosingOps.
  void initializeDomainAndEnclosingOps();

  /// Replace the dim and symbol values that are internally defined SSA values,
  /// which will not be available in the symbol table, with values that can be
  /// found in the symbol table, or more importantly, in the final OpenScop
  /// representation. Recall that every symbol should be found in the OpenScop.
  /// For example, SSA values defined in the callee body won't be found in the
  /// symbol table, thus it should be replaced by values in function arg list
  /// and/or constants. Otherwise, we may construct domain relations that use
  /// values that won't exist in the OpenScop representation. This function
  /// won't have any side effect on the MLIR code. An error will be asserted if
  /// we cannot find a proper replacement for internally defined SSA values.
  void replaceDomainDimAndSymbolValues();

  void getArgsValueMapping(BlockAndValueMapping &argMap);

  /// Name of the callee, as well as the scop.stmt. It will also be the
  /// symbol in the OpenScop representation.
  const llvm::StringRef name;
  /// The caller to the scop.stmt func.
  mlir::CallOp caller;
  /// The scop.stmt callee.
  mlir::FuncOp callee;
  /// The domain of the caller.
  FlatAffineConstraints domain;
  /// Enclosing for/if operations for the caller.
  EnclosingOpList enclosingOps;
};

} // namespace polymer

/// Create ScopStmtImpl from only the caller/callee pair.
std::unique_ptr<ScopStmtImpl> ScopStmtImpl::get(mlir::Operation *callerOp,
                                                mlir::Operation *calleeOp) {
  // We assume that the callerOp is of type mlir::CallOp, and the calleeOp is a
  // mlir::FuncOp. If not, these two cast lines will raise error.
  mlir::CallOp caller = cast<mlir::CallOp>(callerOp);
  mlir::FuncOp callee = cast<mlir::FuncOp>(calleeOp);
  llvm::StringRef name = caller.getCallee();

  // Create the stmt instance.
  auto stmt = std::make_unique<ScopStmtImpl>(name, caller, callee);

  return stmt;
}

void ScopStmtImpl::initializeDomainAndEnclosingOps() {
  // Extract the affine for/if ops enclosing the caller and insert them into the
  // enclosingOps list.
  getEnclosingAffineForAndIfOps(*caller, &enclosingOps);

  // The domain constraints can then be collected from the enclosing ops.
  getIndexSet(enclosingOps, &domain);

  LLVM_DEBUG(llvm::dbgs() << "Initialized domain:\n");
  LLVM_DEBUG(domain.dump());
}

void ScopStmtImpl::getArgsValueMapping(BlockAndValueMapping &argMap) {
  auto callerArgs = caller.getArgOperands();
  auto calleeArgs = callee.getArguments();
  unsigned numArgs = callerArgs.size();

  argMap.clear();
  for (unsigned i = 0; i < numArgs; i++)
    argMap.map(calleeArgs[i], callerArgs[i]);
}

ScopStmt::ScopStmt(Operation *caller, Operation *callee)
    : impl{ScopStmtImpl::get(caller, callee)} {}

ScopStmt::~ScopStmt() = default;
ScopStmt::ScopStmt(ScopStmt &&) = default;
ScopStmt &ScopStmt::operator=(ScopStmt &&) = default;

const FlatAffineConstraints &ScopStmt::getDomain() const {
  return impl->domain;
}

void ScopStmt::getEnclosingOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                               bool forOnly) const {
  for (mlir::Operation *op : impl->enclosingOps)
    if (!forOnly || isa<mlir::AffineForOp>(op))
      ops.push_back(op);
}

mlir::FuncOp ScopStmt::getCallee() const { return impl->callee; }
mlir::CallOp ScopStmt::getCaller() const { return impl->caller; }

void ScopStmt::getIndvars(llvm::SmallVectorImpl<mlir::Value> &indvars) const {
  indvars.clear();

  const FlatAffineConstraints &domain = getDomain();
  domain.getIdValues(0, domain.getNumDimIds(), &indvars);
}

void ScopStmt::updateScatTree(ScatTreeNode &root) const {
  llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
  getEnclosingOps(enclosingOps);

  root.insertPath(enclosingOps, getCaller());
}

void ScopStmt::getScats(const ScatTreeNode &root,
                        llvm::SmallVectorImpl<unsigned> &scats) const {
  llvm::SmallVector<mlir::Operation *, 8> enclosingOps;
  getEnclosingOps(enclosingOps);

  root.getPathIds(enclosingOps, getCaller(), scats);
}

void ScopStmt::getAccesses(
    llvm::SmallVectorImpl<mlir::MemRefAccess> &accesses) const {
  getCallee().walk([&](mlir::Operation *op) {
    if (isa<mlir::AffineWriteOpInterface, mlir::AffineReadOpInterface>(op))
      accesses.push_back(mlir::MemRefAccess(op));
  });
}

void ScopStmt::getAccessMap(const MemRefAccess &access,
                            AffineValueMap &vMap) const {
  // Get the default access map.
  access.getAccessMap(&vMap);

  // Get caller/callee value mappings.
  BlockAndValueMapping argMap;
  impl->getArgsValueMapping(argMap);

  // Replace the values in vMap by the caller operands.
  // Initially, there are cases you cannot replace, e.g., a Value in the vMap is
  // actually the result from an affine.apply, or it is a result from another
  // operation in the callee. We will design additional passes to resolve these
  // issues, and here we simply afssume that every value in vMap is from the
  // callee's BlockArguments, and therefore, they can be replaced directly by
  // the corresponding operands of the caller.
  SmallVector<mlir::Value, 8> replacedOperands;
  for (mlir::Value operand : vMap.getOperands()) {
    assert(mlir::isTopLevelValue(operand) &&
           "Operand of the access value map should be a top-level block "
           "argument.");
    replacedOperands.push_back(argMap.lookup(operand));
  }

  // Reset the values in vMap.
  vMap.reset(vMap.getAffineMap(), replacedOperands);
}

void ScopStmt::getAccessConstraints(const mlir::MemRefAccess &access,
                                    const OslScopSymbolTable &symbolTable,
                                    mlir::FlatAffineConstraints &cst) const {
  getAccessConstraints(access, symbolTable, cst, getDomain());
}

void ScopStmt::getAccessConstraints(const mlir::MemRefAccess &access,
                                    const OslScopSymbolTable &symbolTable,
                                    mlir::FlatAffineConstraints &cst,
                                    mlir::FlatAffineConstraints domain) const {
  // Get caller/callee value mappings.
  BlockAndValueMapping argMap;
  impl->getArgsValueMapping(argMap);

  // Get the affine value map for the access.
  mlir::AffineValueMap vMap;
  getAccessMap(access, vMap);

  cst.reset();
  cst.mergeAndAlignIdsWithOther(0, &domain);

  unsigned numSymbols = vMap.getAffineMap().getNumSymbols();
  cst.setDimSymbolSeparation(numSymbols);

  // The results of the affine value map, which are the access addresses, will
  // be placed to the leftmost of all columns.
  cst.composeMap(&vMap);

  cst.setDimSymbolSeparation(domain.getNumSymbolIds());
  // Add the memref equation.
  mlir::Value memref = argMap.lookup(access.memref);
  cst.addDimId(0, memref);
  cst.setIdToConstant(0, symbolTable.lookupId(memref));
}

void ScopStmt::getAccessMapAndMemRef(mlir::Operation *op,
                                     mlir::AffineValueMap *vMap,
                                     mlir::Value *memref) const {
  assert(op->getParentOfType<mlir::FuncOp>() &&
         "The parent of the given `op` should be a FuncOp (callee)");
  assert(getCallee() == op->getParentOp() &&
         "The parent of the given `op` should be callee.");

  BlockAndValueMapping argMap;
  impl->getArgsValueMapping(argMap);

  // Collect the access AffineValueMap that binds to operands in the callee.
  MemRefAccess access(op);
  getAccessMap(access, *vMap);

  // Set the memref.
  *memref = argMap.lookup(access.memref);
}
