#include "polymer/EmitOpenSCoP.h"

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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace mlir;
using namespace llvm;
using namespace polymer;

#define DEBUG_TYPE "emit-openscop"

namespace {

/// Tree that holds scattering information. This node can represent an induction
/// variable or a statement. A statement is constructed as a leaf node.
class ScatteringTreeNode {
public:
  ScatteringTreeNode(bool isLeaf = false) : isLeaf(isLeaf) {}
  ScatteringTreeNode(mlir::Value iv) : iv(iv), isLeaf(false) {}

  /// Children of the current node.
  std::vector<std::unique_ptr<ScatteringTreeNode>> children;

  /// Mapping from IV to child ID.
  llvm::DenseMap<mlir::Value, unsigned> valueIdMap;

  /// Induction variable.
  mlir::Value iv;

  /// If this node is a statement, then isLeaf is true.
  bool isLeaf;
};

/// Insert a statement characterized by its enclosing operations into a
/// "scattering tree". This is done by iterating through every enclosing for-op
/// from the outermost to the innermost, and we try to traverse the tree by the
/// IVs of these ops. If an IV does not exist, we will insert it into the tree.
/// After that, we insert the current load/store statement into the tree as a
/// leaf. In this progress, we keep track of all the IDs of each child we meet
/// and the final leaf node, which will be used as the scattering.
void insertStatement(ScatteringTreeNode *root,
                     SmallVectorImpl<Operation *> &enclosingOps,
                     SmallVectorImpl<unsigned> &scattering) {
  ScatteringTreeNode *curr = root;

  for (unsigned i = 0, e = enclosingOps.size(); i < e; i++) {
    Operation *op = enclosingOps[i];
    // We only handle for op here.
    // TODO: is it necessary to deal with if?
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      SmallVector<mlir::Value, 4> indices;
      extractForInductionVars(forOp, &indices);

      for (auto iv : indices) {
        auto it = curr->valueIdMap.find(iv);
        if (it != curr->valueIdMap.end()) {
          // Add a new element to the scattering.
          scattering.push_back(it->second);
          // move to the next IV along the tree.
          curr = curr->children[it->second].get();
        } else {
          // No existing node for such IV is found, create a new one.
          auto node = std::make_unique<ScatteringTreeNode>(iv);

          // Then insert the newly created node into the children set, update
          // the value to child ID map, and move the cursor to this new node.
          curr->children.push_back(std::move(node));
          unsigned valueId = curr->children.size() - 1;
          curr->valueIdMap[iv] = valueId;
          scattering.push_back(valueId);
          curr = curr->children.back().get();
        }
      }
    }
  }

  // Append the leaf node for statement
  auto leaf = std::make_unique<ScatteringTreeNode>(/*isLeaf=*/true);
  curr->children.push_back(std::move(leaf));
  scattering.push_back(curr->children.size() - 1);
}

} // namespace

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

  /// Emit OpenSCoP definitions for all functions in the given module.
  void emitMLIRModule(ModuleOp module);

private:
  /// Emit a OpenSCoP definition for a single function.
  LogicalResult emitFuncOp(FuncOp func);
};

LogicalResult ModuleEmitter::emitFuncOp(mlir::FuncOp func) {
  // TODO: for now we assume there is no parameter
  os << "<OpenScop>\n";
  os << R"XXX(
# =============================================== Global
# Backend Language
C
# Context
CONTEXT
0 2 0 0 0 0
# Parameter names are provided
0
)XXX";

  // We iterate through every operations in the function and extrat load/store
  // operations out into loadAndStoreOps.
  SmallVector<Operation *, 8> loadAndStoreOps;
  func.getOperation()->walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << Twine(loadAndStoreOps.size())
                          << " number of load/store operations.\n");
  // Emit number of statements
  os << "# Number of statements\n" << Twine(loadAndStoreOps.size()) << "\n\n";

  // Create the root tree node.
  ScatteringTreeNode root;

  // Maintain the identifiers of memref objects
  llvm::DenseMap<mlir::Value, unsigned> memrefIdMap;

  for (unsigned i = 0, e = loadAndStoreOps.size(); i < e; i++) {
    Operation *op = loadAndStoreOps[i];
    LLVM_DEBUG(op->dump());

    os << "# =============================================== Statement "
       << Twine(i + 1) << "\n";
    // Each statement in the MLIR affine case is a load/store, which means
    // besides the domain and scattering relations, there will be only one
    // additional access relation. In together there will be 3 of them.
    // TODO: check whether there is any missed cases.
    os << "# Number of relations describing the statement\n"
       << Twine(3) << "\n\n";

    // Get the domain first, which is structured as FlatAffineConstraints.
    FlatAffineConstraints domain;
    // TODO: make getOpIndexSet publicly available
    SmallVector<Operation *, 4> ops;
    getEnclosingAffineForAndIfOps(*op, &ops);
    if (failed(getIndexSet(ops, &domain)))
      return failure();
    LLVM_DEBUG(domain.dump());

    os << "# ----------------------------------------------  " << Twine(i + 1)
       << ".1 Domain\n";
    os << "DOMAIN\n";

    // TODO: create a unified API for printing matrices
    // Print the dimensionality of the domain relation matrix.
    os << Twine(domain.getNumConstraints())      // num rows
       << " " << Twine(domain.getNumCols() + 1)  // num columns
       << " " << Twine(domain.getNumDimIds())    // num output dims
       << " " << Twine(0)                        // num input dims (TODO: true?)
       << " " << Twine(domain.getNumLocalIds())  // num local dims
       << " " << Twine(domain.getNumSymbolIds()) // num parameters
       << "\n";
    // Print matrix header comment.
    os << "# e/i |";
    for (unsigned j = 0; j < domain.getNumDimIds(); j++)
      os << " i" << Twine(j);
    os << " |";
    for (unsigned j = 0; j < domain.getNumSymbolIds(); j++)
      os << " N" << Twine(j); // TODO: figure out the actual argument name
    os << " | 1\n";

    // Print every equality.
    for (unsigned j = 0; j < domain.getNumEqualities(); j++) {
      auto eq = domain.getEquality(j);

      os << format_decimal(0, 3) << " "; // indicates it's an equality
      for (unsigned k = 0; k < domain.getNumCols(); k++)
        os << format_decimal(eq[k], 3) << " ";
      os << "\n";
    }
    // Print every inequality.
    for (unsigned j = 0; j < domain.getNumInequalities(); j++) {
      auto inEq = domain.getInequality(j);

      os << format_decimal(1, 3) << " "; // indicates it's an inequality
      for (unsigned k = 0; k < domain.getNumCols(); k++)
        os << format_decimal(inEq[k], 3) << " ";
      os << "\n";
    }
    os << "\n";

    // Get the scattering. By using insertStatement, we create new nodes in the
    // scattering tree representation rooted at `root`, and get the result
    // scattering relation in the `scattering` vector.
    // TODO: consider strided loop indices.
    SmallVector<unsigned, 8> scattering;
    insertStatement(&root, ops, scattering);

    os << "# ----------------------------------------------  " << Twine(i + 1)
       << ".2 Scattering\n";
    os << "SCATTERING\n";

    // TODO: can we wrap these calculation into a bigger data structure like
    // FlatAffineConstraints?
    // Elements (N of them) in `scattering` are constants, and there are IVs
    // interleaved them. Therefore, we have 2N - 1 number of scattering
    // equalities.
    unsigned numScatteringEqualities = scattering.size() * 2 - 1;
    // Columns include new scattering dimensions and those from the domain.
    unsigned numScatteringCols =
        numScatteringEqualities + domain.getNumCols() + 1;

    os << numScatteringEqualities         // num rows
       << " " << numScatteringCols        // num columns
       << " " << numScatteringEqualities  // num scattering dimensions
       << " " << domain.getNumDimIds()    // num iterators
       << " " << domain.getNumLocalIds()  // num local iterators
       << " " << domain.getNumSymbolIds() // num parameters
       << "\n";
    // Print matrix header comment.
    os << "# e/i |";
    for (unsigned j = 0; j < numScatteringEqualities; j++)
      os << " s" << Twine(j);
    os << " |";
    for (unsigned j = 0; j < domain.getNumDimIds(); j++)
      os << " i" << Twine(j);
    os << " |";
    for (unsigned j = 0; j < domain.getNumSymbolIds(); j++)
      os << " N" << Twine(j); // TODO: figure out the actual argument name
    os << " | 1\n";

    // Print every equality.
    // Each time we set the equality for one scattering dimension. If j is odd,
    // we are setting the constant; otherwise, it will be set to a loop IV.
    for (unsigned j = 0; j < numScatteringEqualities; j++) {
      os << format_decimal(0, 3) << " "; // indicates it's an equality
      for (unsigned k = 0; k < numScatteringEqualities; k++)
        os << format_decimal(-static_cast<int64_t>(k == j), 3) << " ";
      for (unsigned k = 0; k < domain.getNumDimIds(); k++) {
        if (j % 2) // odd, set the scattering dimension to the loop IV
          os << format_decimal(k == (j / 2), 3) << " ";
        else
          os << format_decimal(0, 3) << " ";
      }
      for (unsigned k = 0;
           k < domain.getNumLocalIds() + domain.getNumSymbolIds(); k++)
        os << format_decimal(0, 3) << " ";
      if (j % 2 == 0) // even, set the constant
        os << format_decimal(scattering[j / 2], 3) << " ";
      else
        os << format_decimal(0, 3) << " ";

      os << "\n";
    }
    os << "\n";

    // Get the access
    MemRefAccess access(op);
    auto it = memrefIdMap.find(access.memref);
    if (it == memrefIdMap.end())
      memrefIdMap[access.memref] = memrefIdMap.size() + 1;
    auto memrefId = memrefIdMap[access.memref];

    AffineValueMap accessMap;
    access.getAccessMap(&accessMap);
    std::vector<SmallVector<int64_t, 8>> flatExprs;
    FlatAffineConstraints localVarCst;

    if (failed(getFlattenedAffineExprs(accessMap.getAffineMap(), &flatExprs,
                                       &localVarCst)))
      return failure();

    assert(flatExprs.size() > 0 &&
           "Number of flat expressions should be larger than 0.");

    LLVM_DEBUG(llvm::dbgs()
               << "Number of flat exprs: " << flatExprs.size() << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "Flat expr size: " << flatExprs[0].size() << "\n");

    // Number of equalities equals to the number of enclosing loop indices
    // plus 1 (the array itself).
    unsigned numAccessEqualities = domain.getNumDimIds() + 1;
    unsigned numAccessCols = domain.getNumCols() + numAccessEqualities + 1;
    unsigned numFlatExprCols = flatExprs[0].size();

    os << "# ----------------------------------------------  " << Twine(i + 1)
       << ".3 Access\n";
    os << (access.isStore() ? "WRITE" : "READ") << "\n";

    os << numAccessEqualities             // num rows
       << " " << numAccessCols            // num columns
       << " " << numAccessEqualities      // num scattering dimensions
       << " " << domain.getNumDimIds()    // num iterators
       << " " << domain.getNumLocalIds()  // num local iterators
       << " " << domain.getNumSymbolIds() // num parameters
       << "\n";
    // Print matrix header comment.
    os << "# e/i | Arr";
    for (unsigned j = 0; j < numAccessEqualities - 1; j++)
      os << " [i" << Twine(j) << "]";
    os << " |";
    for (unsigned j = 0; j < domain.getNumDimIds(); j++)
      os << " i" << Twine(j);
    os << " |";
    for (unsigned j = 0; j < domain.getNumSymbolIds(); j++)
      os << " N" << Twine(j); // TODO: figure out the actual argument name
    os << " | 1\n";

    // Print equalities
    for (unsigned j = 0; j < numAccessEqualities; j++) {
      os << format_decimal(0, 3) << " "; // indicates it's an equality

      for (unsigned k = 0; k < numAccessEqualities; k++)
        os << format_decimal(-static_cast<int64_t>(k == j), 3) << " ";

      if (j == 0) {
        for (unsigned k = 0; k < domain.getNumCols() - 1; k++)
          os << format_decimal(0, 3) << " ";
        os << format_decimal(memrefId, 3) << " ";
      } else {
        // TODO: consider local variables.
        for (unsigned k = 0; k < numFlatExprCols; k++) {
          if (k == numFlatExprCols - 1)
            os << format_decimal(0, 3) << " ";

          os << format_decimal(flatExprs[j - 1][k], 3) << " ";
        }
      }

      os << "\n";
    }
    os << "\n";

    // We don't consider statement extension here.
    // TODO: consider statement extensions later.
    os << R"XXX(
# ----------------------------------------------  Statement Extensions
# Number of Statement Extensions
0
)XXX";
  }

  os << "</OpenScop>\n";

  return success();
}

/// The entry function to the current OpenSCoP emitter.
void ModuleEmitter::emitMLIRModule(ModuleOp module) {
  // Emit a single OpenSCoP definition for each function.
  for (auto &op : *module.getBody()) {
    if (auto func = dyn_cast<mlir::FuncOp>(op)) {
      if (failed(emitFuncOp(func))) {
        state.encounteredError = true;
        return;
      }
    }
  }
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