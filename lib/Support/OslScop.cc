//===- OslScop.cc -----------------------------------------------*- C++ -*-===//
//
// This file implements the C++ wrapper for the Scop struct in OpenScop.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/OslScop.h"

#include "osl/osl.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Support/LogicalResult.h"

#include <vector>

using namespace polymer;
using namespace mlir;

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
static void getConstraintRows(FlatAffineConstraints &cst,
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

void OslScop::createStatement() {
  osl_statement_p stmt = osl_statement_malloc();
  osl_statement_add(&(scop->statement), stmt);
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

void OslScop::addRelation(int target, int type, int numRows, int numCols,
                          int numOutputDims, int numInputDims, int numLocalDims,
                          int numParams, llvm::ArrayRef<int64_t> eqs,
                          llvm::ArrayRef<int64_t> inEqs) {
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

void OslScop::addContextRelation(FlatAffineConstraints cst) {
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
  addRelation(0, OSL_TYPE_CONTEXT, numRows, numCols, 0, 0, 0,
              cst.getNumSymbolIds(), eqs, inEqs);
}

void OslScop::addGeneric(int target, llvm::StringRef tag,
                         llvm::StringRef content) {

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
