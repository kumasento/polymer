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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

#define DEBUG_TYPE "osl-scop"

using namespace polymer;
using namespace mlir;
using namespace llvm;

/// ----------------------------- OslScopSymbolTable ---------------------------

const OslScopSymbolTable::Container &
OslScopSymbolTable::getSymbolTable() const {
  return symbolTable;
}

llvm::StringRef OslScopSymbolTable::lookup(mlir::Value value) const {
  return lookup(value, nullptr);
}

llvm::StringRef OslScopSymbolTable::lookup(mlir::Value value,
                                           unsigned *numSymbolsOfType) const {
  if (getType(value) == SymbolType::NOT_A_SYMBOL)
    return llvm::StringRef("");

  llvm::StringRef prefix = getPrefix(value);
  if (numSymbolsOfType != nullptr)
    *numSymbolsOfType = 0;

  for (const auto &it : symbolTable) {
    if (numSymbolsOfType != nullptr && it.first().startswith(prefix))
      (*numSymbolsOfType)++;
    if (it.second == value)
      return it.first();
  }
  return llvm::StringRef("");
}

mlir::Value OslScopSymbolTable::lookup(llvm::StringRef symbol) const {
  return symbolTable.lookup(symbol);
}

int64_t OslScopSymbolTable::lookupId(mlir::Value value) const {
  llvm::StringRef symbol = lookup(value);
  assert(!symbol.empty());

  int64_t id;
  dropPrefix(symbol, getType(value)).getAsInteger(64, id);

  return id;
}

llvm::StringRef OslScopSymbolTable::lookupOrCreate(mlir::Value value) {
  if (getType(value) == SymbolType::NOT_A_SYMBOL)
    return llvm::StringRef("");

  LLVM_DEBUG(llvm::dbgs() << "getOrCreateSymbol for value: " << value << "\n");
  unsigned numSymbolsOfType;

  llvm::StringRef prefix = getPrefix(value);
  llvm::StringRef foundSymbol = lookup(value, &numSymbolsOfType);
  if (!foundSymbol.empty())
    return foundSymbol;

  // If the symbol doesn't exist, we create a new one following the
  // corresponding prefix and an integral ID that is increased every creation.
  std::string symbol = prefix.str() + std::to_string(numSymbolsOfType + 1);
  LLVM_DEBUG(llvm::dbgs() << "Symbol created: " << symbol << "\n");

  auto result = symbolTable.insert(std::make_pair(symbol, value));
  assert(result.second && "Insertion of the new symbol should be successful.");

  // We return the ref to the key stored in the StringMap, instead of the
  // temporary string object created here.
  return result.first->first();
}

void OslScopSymbolTable::insert(llvm::StringRef symbol, mlir::Value value) {
  symbolTable.insert(std::make_pair(symbol, value));
}

void OslScopSymbolTable::erase(llvm::StringRef symbol) {
  symbolTable.erase(symbol);
}

llvm::StringRef
OslScopSymbolTable::getPrefix(OslScopSymbolTable::SymbolType type) {
  switch (type) {
  case MEMREF:
    return llvm::StringRef("A");
  case INDVAR:
    return llvm::StringRef("i");
  case PARAMETER:
    return llvm::StringRef("P");
  case CONSTANT:
    return llvm::StringRef("C");
  default:
    assert(false && "Given type doesn't have a corresponding prefix.");
  }
}

llvm::StringRef OslScopSymbolTable::getPrefix(mlir::Value value) {
  return getPrefix(getType(value));
}

OslScopSymbolTable::SymbolType
OslScopSymbolTable::getType(llvm::StringRef symbol) {
  for (int i = MEMREF; i != CONSTANT; i++) {
    SymbolType type = static_cast<SymbolType>(i);
    if (symbol.startswith(getPrefix(type)))
      return type;
  }
  return NOT_A_SYMBOL;
}

OslScopSymbolTable::SymbolType OslScopSymbolTable::getType(mlir::Value value) {
  if (mlir::isForInductionVar(value))
    return INDVAR;
  else if (value.getType().isa<mlir::MemRefType>())
    return MEMREF;
  // TODO: it is likely that more values will be marked as parameters than
  // expected. Not sure whether this could cause any correctness issues.
  else if (mlir::isValidSymbol(value))
    return PARAMETER;
  else if (value.getDefiningOp<mlir::ConstantOp>())
    return CONSTANT;
  return NOT_A_SYMBOL;
}

llvm::StringRef OslScopSymbolTable::dropPrefix(llvm::StringRef symbol,
                                               SymbolType type) {
  llvm::StringRef prefix = getPrefix(type);
  return symbol.drop_front(prefix.size());
}

/// ----------------------------- OslScopStmtMap ------------------------------

const OslScopStmtMap::Symbols &OslScopStmtMap::getKeys() const { return keys; }

const ScopStmt &OslScopStmtMap::lookup(llvm::StringRef key) const {
  const auto &it = map.find(key);
  assert(it != map.end() && "Cannot find the given key in OslScopStmtMap.");

  return it->getValue();
}

void OslScopStmtMap::insert(ScopStmt scopStmt) {
  llvm::StringRef symbol = scopStmt.getCallee().getName();
  assert(map.find(symbol) == map.end() &&
         "Shouldn't insert the ScopStmts of the same symbol multiple times.");

  auto result = map.insert(std::make_pair(symbol, std::move(scopStmt)));

  // Here we use the StringRef to the key in the map, which will be persist
  // during the lifespan of OslScop.
  keys.push_back(result.first->first());
}

void OslScopStmtMap::insert(mlir::CallOp caller, mlir::FuncOp callee) {
  insert(ScopStmt(caller, callee));
}

FlatAffineConstraints OslScopStmtMap::getContext() const {
  FlatAffineConstraints ctx;

  // Union with the domains of all Scop statements. We first merge and align the
  // IDs of the context and the domain of the scop statement, and then append
  // the constraints from the domain to the context. Note that we don't want to
  // mess up with the original domain at this point. Trivial redundant
  // constraints will be removed.
  for (const auto &key : keys) {
    const ScopStmt &scopStmt = lookup(key);
    FlatAffineConstraints domain(scopStmt.getDomain());
    FlatAffineConstraints cst(domain);

    ctx.mergeAndAlignIdsWithOther(0, &cst);
    ctx.append(cst);
    ctx.removeRedundantConstraints();
  }

  return ctx;
}

/// ----------------------------- OslScop -------------------------------------

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

/// --------------------------- Constructors -----------------------------------

OslScop::OslScop()
    : scop(osl_scop_unique_ptr{osl_scop_malloc(), osl_scop_free}) {
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

unsigned OslScop::getNumDimsOfArray(unsigned id) const {
  osl_statement *stmt = scop->statement;
  unsigned numDims = 0;

  bool isSet = false;
  while (stmt != nullptr) {
    osl_relation_list *rel_list = stmt->access;
    while (rel_list != nullptr) {
      osl_relation *rel = rel_list->elt;

      for (unsigned i = 0; i < rel->nb_rows; i++) {
        if (osl_int_mone(osl_util_get_precision(), rel->m[i][1]) &&
            osl_int_get_si(osl_util_get_precision(),
                           rel->m[i][rel->nb_columns - 1]) == id) {
          if (!isSet) {
            numDims = rel->nb_output_dims - 1;
            isSet = true;
          } else
            assert((numDims == rel->nb_output_dims - 1) &&
                   "Number of dims should be the same across all access "
                   "relations");

          break;
        }
      }

      rel_list = rel_list->next;
    }

    stmt = stmt->next;
  }

  return numDims;
}

/// --------------------------- Statements -------------------------------------

osl_statement *OslScop::createStatement() {
  osl_statement *stmt = osl_statement_malloc();
  osl_statement_add(&(scop->statement), stmt);

  return stmt;
}

unsigned OslScop::getStatementId(const osl_statement *stmt) const {
  osl_statement *curr = scop->statement;
  unsigned index = 0;

  while (curr) {
    if (curr == stmt)
      return index;

    curr = curr->next;
    index++;
  }

  assert(false && "The given statement cannot be found.");
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
static osl_relation *addRelation(int target, int type, int numRows, int numCols,
                                 int numOutputDims, int numInputDims,
                                 int numLocalDims, int numParams,
                                 llvm::ArrayRef<int64_t> eqs,
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

  return rel;
}

void OslScop::addRelation(int target, int type, int numRows, int numCols,
                          int numOutputDims, int numInputDims, int numLocalDims,
                          int numParams, llvm::ArrayRef<int64_t> eqs,
                          llvm::ArrayRef<int64_t> inEqs) {
  ::addRelation(target, type, numRows, numCols, numOutputDims, numInputDims,
                numLocalDims, numParams, eqs, inEqs, scop);
}

osl_relation *OslScop::addContextRelation(const FlatAffineConstraints &cst) {
  assert(cst.getNumDimIds() == 0 &&
         "Context constraints shouldn't contain dim IDs.");
  assert(cst.getNumLocalIds() == 0 &&
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
  return ::addRelation(0, OSL_TYPE_CONTEXT, numRows, numCols, 0, 0, 0,
                       cst.getNumSymbolIds(), eqs, inEqs, scop);
}

osl_relation *
OslScop::addDomainRelation(const osl_statement *stmt,
                           const mlir::FlatAffineConstraints &cst) {
  return addDomainRelation(getStatementId(stmt), cst);
}

osl_relation *OslScop::addDomainRelation(int stmtId,
                                         const FlatAffineConstraints &cst) {
  SmallVector<int64_t, 8> eqs, inEqs;
  getConstraintRows(cst, eqs);
  getConstraintRows(cst, inEqs, /*isEq=*/false);

  return ::addRelation(stmtId + 1, OSL_TYPE_DOMAIN, cst.getNumConstraints(),
                       cst.getNumCols() + 1, cst.getNumDimIds(), 0,
                       cst.getNumLocalIds(), cst.getNumSymbolIds(), eqs, inEqs,
                       scop);
}

/// Create equations from the list of scat values.
///
/// Unlike the format specified by OpenScop, the given `scats` is an array of
/// unsigned integers collected from scat tree path IDs, which will look like
/// (0, 0) for (0, i, 0), (1, 2, 0) for (1, i, 2, j, 0), and (0) for (0).
/// We assume that the loop IVs are inserted in the same order as those in the
/// given domain.
///
/// Internally, the 1-D eqs array represents an M x N scattering matrix, each
/// row of it is a equation, its columns are listed as [ results, IVs, params, 1
/// ]. The total number of IVs should be the same as the number of dims in the
/// domain, and the results should equal to M, the total number of equations,
/// which should be the sum of .
static void getEqsFromScats(llvm::ArrayRef<unsigned> scats,
                            const mlir::FlatAffineConstraints &domain,
                            std::vector<int64_t> &eqs, unsigned &numScatEqs,
                            unsigned &numScatCols) {
  // Sanity checks.
  assert(scats.size() == 1 + domain.getNumDimIds() &&
         "The number of IVs + 1 should be the size of scats");

  // Elements (N of them) in `scattering` are constants, and there are IVs
  // interleaving them. Therefore, we have 2N - 1 number of scattering
  // equalities.
  numScatEqs = 2 * scats.size() - 1;
  unsigned numOutputCols = numScatEqs; // an alias.
  // Columns include new scattering dimensions and those from the domain.
  numScatCols = numOutputCols + domain.getNumCols();
  unsigned numIndvarCols = domain.getNumDimIds();
  unsigned numParameterCols =
      domain.getNumLocalIds() + domain.getNumSymbolIds();

  // Initialize contents for equalities.
  eqs.resize(numScatEqs * numScatCols);

  for (unsigned row = 0; row < numScatEqs; row++) {
    unsigned startId = row * numScatCols;

    // Output Dims: reuslts.
    // Initializing scattering dimensions by setting the diagonal to -1.
    for (unsigned col = 0; col < numOutputCols; col++)
      eqs[startId + col] = -static_cast<int64_t>(col == row);

    // Input Dims: loop IVs.
    // Relating the loop IVs to the scattering dimensions. If it's the **odd**
    // (row % 2 == 1) equality, set its scattering dimension to 1 if the loop IV
    // matches (the loop IV id, `col`, equals to the equation ID, row, divided
    // by 2, which can be derived easily from the convention of scats);
    // otherwise, it's scattering dimension will be set in the following
    // constant section.
    startId += numOutputCols;
    for (unsigned col = 0; col < numIndvarCols; col++)
      eqs[startId + col] = (row % 2) ? (col == (row / 2)) : 0;

    // Parameters.
    // TODO: consider the parameters that may appear in the scattering
    // dimension.
    startId += numIndvarCols;
    for (unsigned col = 0; col < numParameterCols; col++)
      eqs[startId + col] = 0;

    // Constants.
    // Relating the constants (the last column) to the scattering dimensions.
    // The value to set can be found in scats.
    startId += numParameterCols;
    eqs[startId] = (row % 2) ? 0 : scats[row / 2];
  }
}

osl_relation *
OslScop::addScatteringRelation(const osl_statement *stmt,
                               const mlir::FlatAffineConstraints &domain,
                               llvm::ArrayRef<unsigned> scats) {
  return addScatteringRelation(getStatementId(stmt), domain, scats);
}

osl_relation *
OslScop::addScatteringRelation(int stmtId,
                               const mlir::FlatAffineConstraints &domain,
                               llvm::ArrayRef<unsigned> scats) {
  unsigned numScatEqs, numScatCols;

  // Create equalities and inequalities. Note that there is no inequality for
  // scattering relations.
  std::vector<int64_t> eqs, inEqs;
  getEqsFromScats(scats, domain, eqs, numScatEqs, numScatCols);

  // Then put them into the scop as a SCATTERING relation.
  return ::addRelation(stmtId + 1, OSL_TYPE_SCATTERING, numScatEqs,
                       numScatCols + 1, numScatEqs, domain.getNumDimIds(),
                       domain.getNumLocalIds(), domain.getNumSymbolIds(), eqs,
                       inEqs, scop);
}

osl_relation *OslScop::addAccessRelation(const osl_statement *stmt, bool isRead,
                                         const FlatAffineConstraints &domain,
                                         const FlatAffineConstraints &cst) {
  return addAccessRelation(getStatementId(stmt), isRead, domain, cst);
}

osl_relation *OslScop::addAccessRelation(int stmtId, bool isRead,
                                         const FlatAffineConstraints &domain,
                                         const FlatAffineConstraints &cst) {
  SmallVector<int64_t, 8> eqs, inEqs;
  getConstraintRows(cst, eqs);
  getConstraintRows(cst, inEqs, /*isEq=*/false);

  // We need to reverse the sign otherwise we may trigger issues in Pluto.
  for (unsigned i = 0; i < eqs.size(); i++)
    eqs[i] = -eqs[i];

  // Then put them into the scop as an ACCESS relation.
  // We find the number of output dims by counting how many consecutive dim IDs
  // don't have Value associated.
  unsigned numOutputDims = 1;
  for (unsigned pos = 1; pos < cst.getNumDimIds(); pos++) {
    if (!cst.getId(pos).hasValue())
      numOutputDims++;
    else
      break;
  }
  unsigned numInputDims = cst.getNumDimIds() - numOutputDims;

  return ::addRelation(stmtId + 1, isRead ? OSL_TYPE_READ : OSL_TYPE_WRITE,
                       cst.getNumConstraints(), cst.getNumCols() + 1,
                       numOutputDims, numInputDims, cst.getNumLocalIds(),
                       cst.getNumSymbolIds(), eqs, inEqs, scop);
}

/// --------------------------- Extensions ------------------------------------

/// Add a new generic field to a statement. `target` gives the statement ID.
/// `content` specifies the data field in the generic.
static osl_generic *addGeneric(int target, llvm::StringRef tag,
                               llvm::StringRef content,
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

  return generic;
}

osl_generic *OslScop::addGeneric(int target, llvm::StringRef tag,
                                 llvm::StringRef content) {
  return ::addGeneric(target, tag, content, scop);
}

/// Add an extension generic content to the whole osl_scop. tag specifies what
/// the generic type is.
static osl_generic *addExtensionGeneric(llvm::StringRef tag,
                                        llvm::StringRef content,
                                        OslScop::osl_scop_unique_ptr &scop) {
  return addGeneric(0, tag, content, scop);
}

/// Add a generic content to the beginning of the whole osl_scop. tag
/// specifies the type of the extension. The content has a space separated
/// list of parameter names.
static osl_generic *addParametersGeneric(llvm::StringRef tag,
                                         llvm::StringRef content,
                                         OslScop::osl_scop_unique_ptr &scop) {
  return addGeneric(-1, "strings", content, scop);
}

/// Add a generic to a single statement, which can be <body>, for example. tag
/// specifies what the generic type is.
static osl_generic *addStatementGeneric(int stmtId, llvm::StringRef tag,
                                        llvm::StringRef content,
                                        OslScop::osl_scop_unique_ptr &scop) {
  return addGeneric(stmtId + 1, tag, content, scop);
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

osl_generic *OslScop::addParameterNames(const OslScopSymbolTable &st) {
  std::string body;
  llvm::raw_string_ostream ss(body);

  SmallVector<std::string, 8> names;

  for (const auto &it : st.getSymbolTable())
    if (st.getType(it.first()) == OslScopSymbolTable::PARAMETER)
      names.push_back(std::string(it.first()));

  if (names.empty())
    return nullptr;

  // The lexicographical order ensures that name "Pi" will only be before "Pj"
  // if i <= j.
  std::sort(names.begin(), names.end());
  for (const auto &s : names)
    ss << s << " ";

  return addParametersGeneric("strings", body, scop);
}

osl_strings *OslScop::getParameterNames() const {
  void *data = osl_generic_lookup(scop->parameters, "strings");
  if (data == nullptr)
    return nullptr;

  return reinterpret_cast<osl_strings *>(data);
}

osl_generic *OslScop::addScatnames(const ScatTreeNode &root) {
  std::string body;
  llvm::raw_string_ostream ss(body);

  // (Depth - 2) * 2 + 1
  // e.g., the depth of a scat tree (0, i, 1, j, 2) is 4, (4 - 2) * 2 + 1 = 5.
  unsigned numScatnames = (root.getDepth() - 2) * 2 + 1;
  for (unsigned i = 0; i < numScatnames; i++)
    ss << "c" << (i + 1) << " ";

  return addExtensionGeneric("scatnames", body, scop);
}

osl_strings *OslScop::getScatnames() const {
  void *data = osl_generic_lookup(scop->extension, "scatnames");
  if (data == nullptr)
    return nullptr;

  return reinterpret_cast<osl_strings *>(data);
}

osl_generic *OslScop::addArrays(const OslScopSymbolTable &st) {
  std::string body;
  llvm::raw_string_ostream ss(body);

  unsigned numArraySymbols = 0;
  for (const auto &it : st.getSymbolTable())
    if (st.getType(it.first()) == OslScopSymbolTable::MEMREF) {
      // Numerical ID and its corresponding textual ID.
      ss << it.first().drop_front() << " " << it.first() << " ";
      numArraySymbols++;
    }

  std::string fullBody = std::to_string(numArraySymbols) + " " + body;
  return addExtensionGeneric("arrays", fullBody, scop);
}

osl_arrays *OslScop::getArrays() const {
  void *data = osl_generic_lookup(scop->extension, "arrays");
  if (data == nullptr)
    return nullptr;

  return reinterpret_cast<osl_arrays *>(data);
}

osl_generic *OslScop::addBody(osl_statement *oslStmt, const ScopStmt &scopStmt,
                              const OslScopSymbolTable &st) {
  std::string body;
  llvm::raw_string_ostream ss(body);

  SmallVector<Value, 8> ivs;
  scopStmt.getIndvars(ivs);

  // First output the size of all the induction variables, as well as the
  // symbols for each of them.
  unsigned numIVs = ivs.size();
  ss << numIVs << " ";
  for (Value iv : ivs)
    ss << st.lookup(iv) << " ";
  ss << "\n";

  mlir::FuncOp callee = scopStmt.getCallee();
  mlir::CallOp caller = scopStmt.getCaller();
  ss << callee.getName() << "(";
  interleave(
      caller.getOperands(), ss, [&](Value arg) { ss << st.lookup(arg); }, ",");
  ss << ")";
  return addStatementGeneric(getStatementId(oslStmt), "body", body, scop);
}

osl_generic *OslScop::addFunctionSignature(mlir::FuncOp f,
                                           const OslScopSymbolTable &st) {
  std::string body;
  llvm::raw_string_ostream ss(body);

  ss << f.getName() << "(";
  interleave(
      f.getArguments(), ss, [&](Value arg) { ss << st.lookup(arg); }, ",");
  ss << ")";

  return addExtensionGeneric("strings", body, scop);
}

llvm::StringRef OslScop::getFunctionSignature() const {
  void *generic = osl_generic_lookup(scop->extension, "strings");
  if (generic == nullptr) {
    llvm::errs() << "Cannot find generic of <strings> in scop->extension.";
    return "";
  }

  osl_strings *strings = reinterpret_cast<osl_strings *>(generic);
  assert(strings != nullptr);

  // There should be a single line in it.
  if (osl_strings_size(strings) != 1)
    return "";

  return strings->string[0];
}
