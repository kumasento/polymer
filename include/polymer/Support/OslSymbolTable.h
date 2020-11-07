//===- OslSymbolTable.h -----------------------------------------*- C++ -*-===//
//
// This file declares the OslSymbolTable class that stores the mapping between
// symbols and MLIR values.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_OSLSYMBOLTABLE_H
#define POLYMER_SUPPORT_OSLSYMBOLTABLE_H

#include <unordered_map>

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

using namespace llvm;
using namespace mlir;

namespace mlir {
class Operation;
class Value;
} // namespace mlir

namespace polymer {

class OslScopStmtOpSet;

// TODO: refactorize this data structure.
class OslSymbolTable {
public:
  using OpSet = OslScopStmtOpSet;
  using OpSetPtr = std::unique_ptr<OpSet>;

  enum SymbolType { LoopIV, Memref, StmtOpSet };

  Value getValue(StringRef key);

  OpSet getOpSet(StringRef key);

  void insertOpIntoOpSet(StringRef key, Operation *op);

  void setValue(StringRef key, Value val, SymbolType type);

  void setOpSet(StringRef key, OpSet val, SymbolType type);

  unsigned getNumValues(SymbolType type);

  unsigned getNumOpSets(SymbolType type);

  void getValueSymbols(SmallVectorImpl<StringRef> &symbols);

  void getOpSetSymbols(SmallVectorImpl<StringRef> &symbols);

  // TODO: don't expose this
  llvm::DenseMap<mlir::Value, std::string> ivArgToName;

  std::unordered_map<std::string, std::string> scatNameToIter;

private:
  StringMap<OpSet> nameToStmtOpSet;
  StringMap<Value> nameToLoopIV;
  StringMap<Value> nameToMemref;
};

} // namespace polymer

#endif
