//===- OslScop.cc -----------------------------------------------*- C++ -*-===//
//
// This file implements the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScop.h"

#include "osl/osl.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

using namespace polymer;
using namespace mlir;
using namespace llvm;

/// Create osl_vector from a STL vector. Since the input vector is of type
/// int64_t, we can safely assume the osl_vector we will generate has 64 bits
/// precision. The input vector doesn't contain the e/i indicator.
static void getOslVector(bool isEq, llvm::ArrayRef<int64_t> vec,
                         osl_vector_p *oslVec) {
  *oslVec = osl_vector_pmalloc(64, vec.size() + 1);

  // Set the e/i field.
  osl_int_t val;
  val.dp = isEq ? 0 : 1;
  (*oslVec)->v[0] = val;

  // Set the rest of the vector.
  for (int i = 0, e = vec.size(); i < e; i++) {
    osl_int_t val;
    val.dp = vec[i];
    (*oslVec)->v[i + 1] = val;
  }
}

/// Get the statement given by its index.
static osl_statement_p getOslStatement(osl_scop_p scop, unsigned index) {
  osl_statement_p stmt = scop->statement;
  for (unsigned i = 0; i <= index; i++) {
    // stmt accessed in the linked list before counting to index should not be
    // NULL.
    assert(stmt && "index exceeds the range of statements in scop.");
    if (i == index)
      return stmt;
    stmt = stmt->next;
  }
  return nullptr;
}

/// Get rows from the given FlatAffineConstraints data structure, which can be
/// equalities or inequalities. The content in these rows should be organized in
/// a row-major order.
static void getConstraintRows(const FlatAffineConstraints &cst,
                              SmallVectorImpl<int64_t> &rows,
                              bool isEq = true) {
  unsigned numRows = isEq ? cst.getNumEqualities() : cst.getNumInequalities();
  unsigned numDimIds = cst.getNumDimIds();
  unsigned numLocalIds = cst.getNumLocalIds();
  unsigned numSymbolIds = cst.getNumSymbolIds();

  for (unsigned i = 0; i < numRows; i++) {
    // Get the row based on isEq.
    auto row = isEq ? cst.getEquality(i) : cst.getInequality(i);

    unsigned numCols = row.size();
    if (i == 0)
      rows.resize(numRows * numCols);

    // Dims stay at the same positions.
    for (unsigned j = 0; j < numDimIds; j++)
      rows[i * numCols + j] = row[j];
    // Output local ids before symbols.
    for (unsigned j = 0; j < numLocalIds; j++)
      rows[i * numCols + j + numDimIds] = row[j + numDimIds + numSymbolIds];
    // Output symbols in the end.
    for (unsigned j = 0; j < numSymbolIds; j++)
      rows[i * numCols + j + numDimIds + numLocalIds] = row[j + numDimIds];
    // Finally outputs the constant.
    rows[i * numCols + numCols - 1] = row[numCols - 1];
  }
}

OslScop::OslScop()
    : scop(osl_scop_unique_ptr{osl_scop_malloc(), osl_scop_free}),
      scatTreeRoot(std::make_unique<ScatTreeNode>()) {

  // Additional setup for the language and registry.
  OSL_strdup(scop->language, "C");
  // Use the default interface registry
  osl_interface_p registry = osl_interface_get_default_registry();
  scop->registry = osl_interface_clone(registry);
}

OslScop::~OslScop() {}

void OslScop::print(FILE *fp) const { osl_scop_print(fp, scop.get()); }

bool OslScop::validate() const {
  // TODO: do we need to check the scoplib compatibility?
  return osl_scop_integrity_check(scop.get());
}

/// --------------------------- ScopStmtMap ------------------------------------

const OslScop::ScopStmtMap &OslScop::getScopStmtMap() const {
  return scopStmtMap;
}

void OslScop::addScopStmt(mlir::CallOp caller, mlir::FuncOp callee) {
  llvm::StringRef symbol = callee.getName();
  auto result =
      scopStmtMap.insert(std::make_pair(symbol, ScopStmt(caller, callee)));

  // Here we use the StringRef to the key in the map, which will be persist
  // during the lifespan of OslScop.
  scopStmtSymbols.push_back(result.first->first());
}

/// This function tries to remove constraints that has non-zero coefficients at
/// the given position.
static void removeConstraintsWithNonZeroCoeff(FlatAffineConstraints &cst,
                                              unsigned pos) {
  unsigned row = 0;
  while (row < cst.getNumEqualities() + cst.getNumInequalities()) {
    bool isEq = row < cst.getNumEqualities();
    unsigned id = isEq ? row : row - cst.getNumEqualities();

    int64_t coeff = isEq ? cst.atEq(id, pos) : cst.atIneq(id, pos);
    if (coeff != 0) {
      if (isEq)
        cst.removeEquality(id);
      else
        cst.removeInequality(id);
    } else {
      row++;
    }
  }
}

/// Remove dim columns and all constraints that have non-zero coefficients on
/// those dim values.
static void removeDims(mlir::FlatAffineConstraints &cst) {
  for (unsigned pos = 0; pos < cst.getNumDimIds(); pos++)
    removeConstraintsWithNonZeroCoeff(cst, pos);
  while (cst.getNumDimIds() > 0)
    cst.removeId(0);
}

void OslScop::getContextConstraints(mlir::FlatAffineConstraints &ctx,
                                    bool isRemovingDims) const {
  ctx.reset();

  // Union with the domains of all Scop statements. We first merge and align the
  // IDs of the context and the domain of the scop statement, and then append
  // the constraints from the domain to the context. Note that we don't want to
  // mess up with the original domain at this point. Trivial redundant
  // constraints will be removed.
  for (const auto &it : scopStmtMap) {
    const FlatAffineConstraints &domain = it.second.getDomain();
    FlatAffineConstraints cst(domain);

    ctx.mergeAndAlignIdsWithOther(0, &cst);
    ctx.append(cst);
    ctx.removeRedundantConstraints();

    if (isRemovingDims) {
      removeDims(ctx);
    } else {
      llvm::SmallVector<mlir::Value, 8> dimValues;
      cst.getIdValues(0, cst.getNumDimIds(), &dimValues);
      for (mlir::Value dim : dimValues)
        ctx.projectOut(dim);
    }
  }
}

/// --------------------------- Statements -------------------------------------

osl_statement *OslScop::createStatement() {
  osl_statement *stmt = osl_statement_malloc();
  osl_statement_add(&(scop->statement), stmt);

  return stmt;
}

osl_statement *OslScop::getStatement(unsigned index) const {
  osl_statement_p curr = scop->statement;
  if (!curr)
    return nullptr;

  for (unsigned i = 0; i < index; i++)
    if (!(curr = curr->next))
      return nullptr;

  return curr;
}

unsigned OslScop::getNumStatements() const {
  return osl_statement_number(scop->statement);
}

/// Create a new relation and initialize its contents. The new relation will
/// be created under the scop member.
/// The target here is an index:
/// 1) if it's 0, then it means the context;
/// 2) otherwise, if it is a positive number, it corresponds to a statement of
/// id=(target-1).
static void addRelation(int target, int type, int numRows, int numCols,
                        int numOutputDims, int numInputDims, int numLocalDims,
                        int numParams, llvm::ArrayRef<int64_t> eqs,
                        llvm::ArrayRef<int64_t> inEqs,
                        OslScop::osl_scop_unique_ptr &scop) {
  // Here we preset the precision to 64.
  osl_relation_p rel = osl_relation_pmalloc(64, numRows, numCols);
  rel->type = type;
  rel->nb_output_dims = numOutputDims;
  rel->nb_input_dims = numInputDims;
  rel->nb_local_dims = numLocalDims;
  rel->nb_parameters = numParams;

  // The number of columns in the given equalities and inequalities, which is
  // one less than the number of columns in the OSL representation (missing e/i
  // indicator).
  size_t numColsInEqs = numCols - 1;

  assert(eqs.size() % numColsInEqs == 0 &&
         "Number of elements in the eqs should be an integer multiply if "
         "numColsInEqs\n");
  int numEqs = eqs.size() / numColsInEqs;

  // Replace those allocated vector elements in rel.
  for (int i = 0; i < numRows; i++) {
    osl_vector_p vec;

    if (i >= numEqs) {
      auto inEq = llvm::ArrayRef<int64_t>(&inEqs[(i - numEqs) * numColsInEqs],
                                          numColsInEqs);
      getOslVector(false, inEq, &vec);
    } else {
      auto eq = llvm::ArrayRef<int64_t>(&eqs[i * numColsInEqs], numColsInEqs);
      getOslVector(true, eq, &vec);
    }

    // Replace the vector content of the i-th row by the contents in
    // constraints.
    osl_relation_replace_vector(rel, vec, i);

    // Free the newly allocated vector
    osl_vector_free(vec);
  }

  // Append the newly created relation to a target linked list, or simply set it
  // to a relation pointer, which is indicated by the target argument.
  if (target == 0) {
    // Simply assign the newly created relation to the context field.
    scop->context = rel;
  } else {
    // Get the pointer to the statement.
    osl_statement_p stmt = getOslStatement(scop.get(), target - 1);

    // Depending on the type of the relation, we decide which field of the
    // statement we should set.
    if (type == OSL_TYPE_DOMAIN) {
      stmt->domain = rel;
    } else if (type == OSL_TYPE_SCATTERING) {
      stmt->scattering = rel;
    } else if (type == OSL_TYPE_ACCESS || type == OSL_TYPE_WRITE ||
               type == OSL_TYPE_READ) {
      osl_relation_list_p relList = osl_relation_list_malloc();
      relList->elt = rel;
      osl_relation_list_add(&(stmt->access), relList);
    }
  }
}

void OslScop::addRelation(int target, int type, int numRows, int numCols,
                          int numOutputDims, int numInputDims, int numLocalDims,
                          int numParams, llvm::ArrayRef<int64_t> eqs,
                          llvm::ArrayRef<int64_t> inEqs) {
  ::addRelation(target, type, numRows, numCols, numOutputDims, numInputDims,
                numLocalDims, numParams, eqs, inEqs, scop);
}

void OslScop::addContextRelation(const FlatAffineConstraints &cst) {
  assert(cst.getNumDimIds() == 0 &&
         "Context constraints shouldn't contain dim IDs.");
  assert(cst.getNumSymbolIds() == 0 &&
         "Context constraints shouldn't contain local IDs.");

  SmallVector<int64_t, 8> eqs, inEqs;
  getConstraintRows(cst, eqs);
  getConstraintRows(cst, inEqs, /*isEq=*/false);

  unsigned numCols = 2 + cst.getNumSymbolIds();
  unsigned numEntries = inEqs.size() + eqs.size();
  assert(numEntries % (numCols - 1) == 0 &&
         "Total number of entries should be divisible by the number of columns "
         "(excluding e/i)");

  unsigned numRows = (inEqs.size() + eqs.size()) / (numCols - 1);
  // Create the context relation.
  ::addRelation(0, OSL_TYPE_CONTEXT, numRows, numCols, 0, 0, 0,
                cst.getNumSymbolIds(), eqs, inEqs, scop);
}

void OslScop::addDomainRelation(int stmtId, const FlatAffineConstraints &cst) {
  SmallVector<int64_t, 8> eqs, inEqs;
  getConstraintRows(cst, eqs);
  getConstraintRows(cst, inEqs, /*isEq=*/false);

  ::addRelation(stmtId + 1, OSL_TYPE_DOMAIN, cst.getNumConstraints(),
                cst.getNumCols() + 1, cst.getNumDimIds(), 0,
                cst.getNumLocalIds(), cst.getNumSymbolIds(), eqs, inEqs, scop);
}

/// Create equations from the list of scat values.
static void getEqsFromScats(llvm::ArrayRef<unsigned> scats,
                            const mlir::FlatAffineConstraints &cst,
                            std::vector<int64_t> &eqs) {
  // Elements (N of them) in `scattering` are constants, and there are IVs
  // interleaved them. Therefore, we have 2N - 1 number of scattering
  // equalities.
  unsigned numScatEqs = scats.size() * 2 - 1;
  // Columns include new scattering dimensions and those from the domain.
  unsigned numScatCols = numScatEqs + cst.getNumCols() + 1;

  // Initialize contents for equalities.
  eqs.resize(numScatEqs * (numScatCols - 1));

  for (unsigned j = 0; j < numScatEqs; j++) {
    unsigned startId = j * (numScatCols - 1);
    // Initializing scattering dimensions by setting the diagonal to -1.
    for (unsigned k = 0; k < numScatEqs; k++)
      eqs[startId + k] = -static_cast<int64_t>(k == j);

    // Relating the loop IVs to the scattering dimensions. If it's the odd
    // equality, set its scattering dimension to the loop IV; otherwise, it's
    // scattering dimension will be set in the following constant section.
    for (unsigned k = 0; k < cst.getNumDimIds(); k++)
      eqs[startId + k + numScatEqs] = (j % 2) ? (k == (j / 2)) : 0;

    // TODO: consider the parameters that may appear in the scattering
    // dimension.
    for (unsigned k = 0; k < cst.getNumLocalIds() + cst.getNumSymbolIds(); k++)
      eqs[startId + k + numScatEqs + cst.getNumDimIds()] = 0;

    // Relating the constants (the last column) to the scattering dimensions.
    eqs[startId + numScatCols - 2] = (j % 2) ? 0 : scats[j / 2];
  }
}

void OslScop::addScatteringRelation(
    int stmtId, const mlir::FlatAffineConstraints &cst,
    llvm::ArrayRef<mlir::Operation *> enclosingOps) {
  // First insert the enclosing ops into the scat tree.
  SmallVector<unsigned, 8> scats;
  scatTreeRoot->insertPath(enclosingOps, scats);

  // Create equalities and inequalities.
  std::vector<int64_t> eqs, inEqs;
  getEqsFromScats(scats, cst, eqs);

  unsigned numScatEqs = scats.size() * 2 - 1;
  unsigned numScatCols = numScatEqs + cst.getNumCols() + 1;

  // Then put them into the scop as a SCATTERING relation.
  ::addRelation(stmtId + 1, OSL_TYPE_SCATTERING, numScatEqs, numScatCols,
                numScatEqs, cst.getNumDimIds(), cst.getNumLocalIds(),
                cst.getNumSymbolIds(), eqs, inEqs, scop);
}

/// Use FlatAffineConstraints to represent the access relation, given by the
/// AffineValueMap vMap. Result is returned as cst.
static void getAccessRelationConstraints(mlir::AffineValueMap &vMap,
                                         mlir::Value memref, unsigned memId,
                                         mlir::FlatAffineConstraints &domain,
                                         mlir::FlatAffineConstraints &cst) {
  cst.reset();
  cst.mergeAndAlignIdsWithOther(0, &domain);

  SmallVector<mlir::Value, 8> idValues;
  domain.getAllIdValues(&idValues);
  llvm::SetVector<mlir::Value> idValueSet;
  for (auto val : idValues)
    idValueSet.insert(val);

  for (auto operand : vMap.getOperands())
    if (!idValueSet.contains(operand)) {
      llvm::errs() << "Operand missing: " << operand << "\n";
    }

  // The results of the affine value map, which are the access addresses, will
  // be placed to the leftmost of all columns.
  cst.composeMap(&vMap);

  // Add the memref equation.
  cst.addDimId(0, memref);
  cst.setIdToConstant(0, memId);
}

void OslScop::addAccessRelation(int stmtId, bool isRead, mlir::Value memref,
                                mlir::AffineValueMap &vMap,
                                FlatAffineConstraints &domain) {
  FlatAffineConstraints cst;

  // Create a new dim of memref and set its value to its corresponding ID.
  unsigned memId;
  llvm::StringRef memSym = getOrCreateSymbol(memref);
  memSym.drop_front().getAsInteger(0, memId);

  // Insert the address dims and put constraints in it.
  getAccessRelationConstraints(vMap, memref, memId, domain, cst);

  SmallVector<int64_t, 8> eqs, inEqs;
  // inEqs will be left empty.
  getConstraintRows(cst, eqs);

  // We need to reverse the sign otherwise we may trigger issues in Pluto.
  for (unsigned i = 0; i < eqs.size(); i++)
    eqs[i] = -eqs[i];

  // Then put them into the scop as an ACCESS relation.
  unsigned numOutputDims = cst.getNumConstraints();
  unsigned numInputDims = cst.getNumDimIds() - numOutputDims;
  ::addRelation(stmtId + 1, isRead ? OSL_TYPE_READ : OSL_TYPE_WRITE,
                cst.getNumConstraints(), cst.getNumCols() + 1, numOutputDims,
                numInputDims, cst.getNumLocalIds(), cst.getNumSymbolIds(), eqs,
                inEqs, scop);
}

/// --------------------------- Extensions ------------------------------------

/// Add a new generic field to a statement. `target` gives the statement ID.
/// `content` specifies the data field in the generic.
static void addGeneric(int target, llvm::StringRef tag, llvm::StringRef content,
                       OslScop::osl_scop_unique_ptr &scop) {
  osl_generic_p generic = osl_generic_malloc();

  // Add interface.
  osl_interface_p interface = osl_interface_lookup(scop->registry, tag.data());
  generic->interface = osl_interface_nclone(interface, 1);

  // Add content
  char *buf;
  OSL_malloc(buf, char *, content.size() * sizeof(char));
  OSL_strdup(buf, content.data());
  generic->data = interface->sread(&buf);

  if (target == 0) {
    // Add to Scop extension.
    osl_generic_add(&(scop->extension), generic);
  } else if (target == -1) {
    // Add to Scop parameters.
    osl_generic_add(&(scop->parameters), generic);
  } else {
    // Add to statement.
    osl_statement_p stmt = getOslStatement(scop.get(), target - 1);
    osl_generic_add(&(stmt->extension), generic);
  }
}

void OslScop::addGeneric(int target, llvm::StringRef tag,
                         llvm::StringRef content) {
  ::addGeneric(target, tag, content, scop);
}

void OslScop::addExtensionGeneric(llvm::StringRef tag,
                                  llvm::StringRef content) {
  ::addGeneric(0, tag, content, scop);
}

void OslScop::addParametersGeneric(llvm::StringRef tag,
                                   llvm::StringRef content) {
  ::addGeneric(-1, "strings", content, scop);
}

void OslScop::addStatementGeneric(int stmtId, llvm::StringRef tag,
                                  llvm::StringRef content) {
  ::addGeneric(stmtId + 1, tag, content, scop);
}

/// We determine whether the name refers to a symbol by looking up the parameter
/// list of the scop.
bool OslScop::isSymbol(llvm::StringRef name) {
  osl_generic_p parameters = scop->parameters;
  if (!parameters)
    return false;

  assert(parameters->next == NULL &&
         "Should only exist one parameters generic object.");
  assert(osl_generic_has_URI(parameters, OSL_URI_STRINGS) &&
         "Parameters should be of strings interface.");

  // TODO: cache this result, otherwise we need O(N) each time calling this API.
  osl_strings_p parameterNames =
      reinterpret_cast<osl_strings_p>(parameters->data);
  unsigned numParameters = osl_strings_size(parameterNames);

  for (unsigned i = 0; i < numParameters; i++)
    if (name.equals(parameterNames->string[i]))
      return true;

  return false;
}

osl_generic_p OslScop::getExtension(llvm::StringRef tag) const {
  osl_generic_p ext = scop->extension;
  osl_interface_p interface = osl_interface_lookup(scop->registry, tag.data());

  while (ext) {
    if (osl_interface_equal(ext->interface, interface))
      return ext;
    ext = ext->next;
  }

  return nullptr;
}

void OslScop::addParameterNamesFromSymbolTable() {
  std::string body;
  llvm::raw_string_ostream ss(body);

  SmallVector<std::string, 8> names;

  for (const auto &it : symbolTable)
    if (getSymbolType(it.first()) == PARAMETER)
      names.push_back(std::string(it.first()));

  // The lexicographical order ensures that name "Pi" will only be before "Pj"
  // if i <= j.
  std::sort(names.begin(), names.end());
  for (const auto &s : names)
    ss << s << " ";

  addParametersGeneric("strings", body);
}

void OslScop::addScatnamesExtensionFromScatTree() {
  std::string body;
  llvm::raw_string_ostream ss(body);

  unsigned numScatnames = scatTreeRoot->getDepth();
  numScatnames = (numScatnames - 2) * 2 + 1;
  for (unsigned i = 0; i < numScatnames; i++)
    ss << "c" << (i + 1) << " ";

  addExtensionGeneric("scatnames", body);
}

void OslScop::addArraysExtensionFromSymbolTable() {
  std::string body;
  llvm::raw_string_ostream ss(body);

  unsigned numArraySymbols = 0;
  for (const auto &it : symbolTable)
    if (getSymbolType(it.first()) == MEMREF) {
      // Numerical ID and its corresponding textual ID.
      ss << it.first().drop_front() << " " << it.first() << " ";
      numArraySymbols++;
    }

  std::string fullBody = std::to_string(numArraySymbols) + body;
  addExtensionGeneric("arrays", fullBody);
}

void OslScop::addBodyExtension(int stmtId, const ScopStmt &stmt) {
  // std::string body;
  // llvm::raw_string_ostream ss(body);

  // SmallVector<mlir::Operation *, 8> forOps;
  // stmt.getEnclosingOps(forOps, /*forOnly=*/true);

  // unsigned numIVs = forOps.size();
  // ss << numIVs << " ";

  // llvm::DenseMap<mlir::Value, unsigned> ivToId;
  // for (unsigned i = 0; i < numIVs; i++) {
  //   mlir::AffineForOp forOp = cast<mlir::AffineForOp>(forOps[i]);
  //   // forOp.dump();
  //   ivToId[forOp.getInductionVar()] = i;
  // }

  // for (unsigned i = 0; i < numIVs; i++)
  //   ss << "i" << i << " ";

  // mlir::CallOp caller = stmt.getCaller();
  // mlir::FuncOp callee = stmt.getCallee();
  // ss << "\n" << callee.getName() << "(";

  // SmallVector<std::string, 8> ivs;
  // SetVector<unsigned> visited;
  // for (unsigned i = 0; i < caller.getNumOperands(); i++) {
  //   mlir::Value operand = caller.getOperand(i);
  //   if (ivToId.find(operand) != ivToId.end()) {
  //     ivs.push_back(std::string(formatv("i{0}", ivToId[operand])));
  //     visited.insert(ivToId[operand]);
  //   }
  // }

  // for (unsigned i = 0; i < numIVs; i++)
  //   if (!visited.contains(i)) {
  //     visited.insert(i);
  //     ivs.push_back(std::string(formatv("i{0}", i)));
  //   }

  // for (unsigned i = 0; i < ivs.size(); i++) {
  //   ss << ivs[i];
  //   if (i != ivs.size() - 1)
  //     ss << ", ";
  // }

  // ss << ")";

  // addGeneric(stmtId + 1, "body", body);
}

/// --------------------------- Symbol Table ----------------------------------

llvm::StringRef OslScop::getOrCreateSymbol(mlir::Value value) {
  unsigned numSymbolsOfType = 0;

  llvm::StringRef prefix = getSymbolPrefix(value);
  for (const auto &it : symbolTable) {
    if (it.first().startswith(prefix))
      numSymbolsOfType++;
    if (it.second == value)
      return it.first();
  }

  /// If the symbol doesn't exist, we create a new one following the
  /// corresponding prefix and an integral ID that is increased every creation.
  std::string symbol = prefix.str() + std::to_string(numSymbolsOfType);
  auto result = symbolTable.insert(std::make_pair(symbol, value));
  assert(result.second && "Insertion of the new symbol should be successful.");

  // We return the ref to the key stored in the StringMap, instead of the
  // temporary string object created here.
  return result.first->first();
}

llvm::StringRef OslScop::getSymbolPrefix(OslScop::SymbolType type) const {
  switch (type) {
  case MEMREF:
    return llvm::StringRef("A");
  case INDVAR:
    return llvm::StringRef("i");
  case PARAMETER:
    return llvm::StringRef("P");
  case CONSTANT:
    return llvm::StringRef("C");
  }
}

llvm::StringRef OslScop::getSymbolPrefix(mlir::Value value) const {
  return getSymbolPrefix(getSymbolType(value));
}

OslScop::SymbolType OslScop::getSymbolType(llvm::StringRef symbol) const {
  for (int i = MEMREF; i != CONSTANT; i++) {
    SymbolType type = static_cast<SymbolType>(i);
    if (symbol.startswith(getSymbolPrefix(type)))
      return type;
  }

  assert(false && "Given symbol doesn't match and prefix.");
}

OslScop::SymbolType OslScop::getSymbolType(mlir::Value value) const {
  if (mlir::isForInductionVar(value))
    return INDVAR;
  else if (value.isa<mlir::BlockArgument>())
    return PARAMETER;
  else if (value.getType().isa<mlir::MemRefType>())
    return MEMREF;
  else if (value.getDefiningOp<mlir::ConstantOp>())
    return CONSTANT;

  assert(false && "Given value cannot correspond to a symbol.");
}
