//===- ConvertFromOpenScop.h ------------------------------------*- C++ -*-===//
//
// This file implements the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#include "cloog/cloog.h"
#include "osl/osl.h"

#include "polymer/Support/OslScop.h"
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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Translation.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

using namespace polymer;
using namespace mlir;

typedef llvm::StringMap<mlir::Operation *> StmtOpMap;
typedef llvm::StringMap<mlir::Value> NameValueMap;

namespace {

/// Build AffineExpr from a clast_expr.
/// TODO: manage the priviledge.
class AffineExprBuilder {
public:
  AffineExprBuilder(MLIRContext *context, std::unique_ptr<OslScop> &scop,
                    CloogOptions *options)
      : b(context), context(context), scop(scop), options(options) {
    reset();
  }

  void process(clast_expr *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);

  void reset();

  void process(clast_name *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);
  void process(clast_term *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);
  void process(clast_binary *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs);
  void process(clast_reduction *expr,
               llvm::SmallVectorImpl<AffineExpr> &affExprs);

  void processSumReduction(clast_reduction *expr,
                           llvm::SmallVectorImpl<AffineExpr> &affExprs);
  void processMinOrMaxReduction(clast_reduction *expr,
                                llvm::SmallVectorImpl<AffineExpr> &affExprs);

  /// OpBuilder used to create AffineExpr.
  OpBuilder b;
  /// The MLIR context
  MLIRContext *context;
  /// The OslScop of the whole program.
  std::unique_ptr<OslScop> &scop;
  ///
  CloogOptions *options;

  llvm::SmallVector<llvm::StringRef, 4> symbolNames;
  llvm::SmallVector<llvm::StringRef, 4> dimNames;
};
} // namespace

void AffineExprBuilder::reset() {
  symbolNames.clear();
  dimNames.clear();
}

/// Get the int64_t representation of a cloog_int_t.
static void getI64(cloog_int_t num, int64_t *res) {
  // TODO: is there a better way to work around this file-based interface?
  // First, we read the cloog integer into a char buffer.
  char buf[100]; // Should be sufficient for int64_t in string.
  FILE *bufFile = fmemopen(reinterpret_cast<void *>(buf), 32, "w");
  cloog_int_print(bufFile, num);
  fclose(bufFile); // Should close the file or the buf won't be flushed.

  // Then we parse the string as int64_t.
  *res = strtoll(buf, NULL, 10);
}

void AffineExprBuilder::process(clast_expr *expr,
                                llvm::SmallVectorImpl<AffineExpr> &affExprs) {

  switch (expr->type) {
  case clast_expr_name:
    process(reinterpret_cast<clast_name *>(expr), affExprs);
    break;
  case clast_expr_term:
    process(reinterpret_cast<clast_term *>(expr), affExprs);
    break;
  case clast_expr_bin:
    process(reinterpret_cast<clast_binary *>(expr), affExprs);
    break;
  case clast_expr_red:
    process(reinterpret_cast<clast_reduction *>(expr), affExprs);
    break;
  }
}

/// Find the name in the scop to determine the type (dim or symbol). The
/// position is decided by the size of dimNames/symbolNames.
/// TODO: handle the dim case.
void AffineExprBuilder::process(clast_name *expr,
                                llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  if (llvm::StringRef(expr->name).startswith("P")) {
    affExprs.push_back(b.getAffineSymbolExpr(symbolNames.size()));
    symbolNames.push_back(expr->name);
  } else {
    llvm::errs()
        << expr->name
        << " is not a valid name can be found as a symbol or a loop IV.\n";
    assert(false);
  }
}

void AffineExprBuilder::process(clast_term *expr,
                                llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  // First get the I64 representation of a cloog int.
  int64_t constant;
  getI64(expr->val, &constant);

  // Next create a constant AffineExpr.
  AffineExpr affExpr = b.getAffineConstantExpr(constant);

  // If var is not NULL, it means this term is var * val. We should create the
  // expr that denotes var and multiplies it with the AffineExpr for val.
  if (expr->var) {
    SmallVector<AffineExpr, 1> varAffExprs;
    process(expr->var, varAffExprs);
    assert(varAffExprs.size() == 1 &&
           "There should be a single expression that stands for the var expr.");

    affExpr = affExpr * varAffExprs[0];
  }

  affExprs.push_back(affExpr);
}

void AffineExprBuilder::process(clast_binary *expr,
                                llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  // Handle the LHS expression.
  SmallVector<AffineExpr, 1> lhsAffExprs;
  process(expr->LHS, lhsAffExprs);
  assert(lhsAffExprs.size() == 1 &&
         "There should be a single LHS affine expr.");

  // Handle the RHS expression, which is an integer constant.
  int64_t rhs;
  getI64(expr->RHS, &rhs);
  AffineExpr rhsAffExpr = b.getAffineConstantExpr(rhs);

  AffineExpr affExpr;

  switch (expr->type) {
  case clast_bin_fdiv:
    affExpr = lhsAffExprs[0].floorDiv(rhsAffExpr);
    break;
  case clast_bin_cdiv:
  case clast_bin_div:
    affExpr = lhsAffExprs[0].ceilDiv(rhsAffExpr);
    break;
  case clast_bin_mod:
    affExpr = lhsAffExprs[0] % rhsAffExpr;
    break;
  }

  affExprs.push_back(affExpr);
}

void AffineExprBuilder::process(clast_reduction *expr,
                                llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  if (expr->n == 1) {
    process(expr->elts[0], affExprs);
    return;
  }

  switch (expr->type) {
  case clast_red_sum:
    processSumReduction(expr, affExprs);
    break;
  case clast_red_min:
  case clast_red_max:
    processMinOrMaxReduction(expr, affExprs);
    break;
  }
}

void AffineExprBuilder::processSumReduction(
    clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  assert(expr->n >= 1 && "Number of reduction elements should be non-zero.");
  assert(expr->elts[0]->type == clast_expr_term &&
         "The first element should be a term.");

  // Build the reduction expression.
  unsigned numAffExprs = affExprs.size();
  process(expr->elts[0], affExprs);
  assert(numAffExprs + 1 == affExprs.size() &&
         "A single affine expr should be appended after processing an expr in "
         "reduction.");

  SmallVector<AffineExpr, 1> currExprs;
  for (int i = 1; i < expr->n; ++i) {
    assert(expr->elts[i]->type == clast_expr_term &&
           "Each element in the reduction list should be a term.");

    clast_term *term = reinterpret_cast<clast_term *>(expr->elts[i]);
    process(term, currExprs);
    assert(currExprs.size() == 1 &&
           "There should be one affine expr corresponds to a single term.");

    // TODO: deal with negative terms.
    // numAffExprs is the index for the current affExpr, i.e., the newly
    // appended one from processing expr->elts[0].
    affExprs[numAffExprs] = affExprs[numAffExprs] + currExprs[0];
  }
}

void AffineExprBuilder::processMinOrMaxReduction(
    clast_reduction *expr, llvm::SmallVectorImpl<AffineExpr> &affExprs) {
  process(expr->elts[0], affExprs);

  for (int i = 1; i < expr->n; i++)
    process(expr->elts[i], affExprs);
}

namespace {

/// Import MLIR code from the CLooG AST.
class Importer {
public:
  Importer(MLIRContext *context, ModuleOp module,
           std::unique_ptr<OslScop> &scop, OslScopSymbolTable &st,
           CloogOptions *options)
      : b(context), context(context), module(module), scop(scop), st(st),
        options(options) {
    b.setInsertionPointToStart(module.getBody());
  }

  /// The main entry for processing the CLooG AST.
  void import(clast_stmt *s);

  mlir::Operation *getFunc() { return f; }

private:
  /// Translate the root statement as a function. The name and signature of the
  /// function is given by the <strings> tag in the OpenScop representation.
  void processStmt(clast_root *rootStmt);

  void processStmt(clast_for *forStmt);
  void processStmt(clast_user_stmt *userStmt);

  /// The given clast_expr specifies a loop bound. It could contain multiple sub
  /// expressions, which will be "merged" by either min for upper bounds, or max
  /// for lower bounds. `operands` and `affMap` are results of this function.
  /// `isUpper` specifies whether we are getting an upper bound or a lower
  /// bound.
  void getAffineLoopBound(clast_expr *expr,
                          llvm::SmallVectorImpl<mlir::Value> &operands,
                          AffineMap &affMap, bool isUpper = false);

  /// Function signature should look like foo(A1, P1).
  void parseFunctionSignature(llvm::StringRef body, std::string &calleeName,
                              llvm::SmallVectorImpl<std::string> &argNames);

  void initParamSymbols(osl_strings *);
  void initArraySymbols(osl_arrays *);

  /// From the given OpenScop representation, we try to iterate through the
  /// symbol table to see if the symbols in <arrays> and scop->parameters
  /// already exists. If not, we will create values for them. To create memref
  /// values (for <arrays>), we will go through the access relations in all
  /// statements, to see if there is any access to the arrays to be created, and
  /// if so, what the dimensions of those values. We also need to make sure the
  /// number of dimensions are consistent. For parameters, we just create index
  /// values for them. All these values will be created as BlockArguments to the
  /// provided FuncOp.
  void initSymbols();

  /// Functions are always inserted before the module terminator.
  Block::iterator getFuncInsertPt() {
    return std::prev(module.getBody()->end());
  }

  /// The current builder, pointing at where the next Instruction should be
  /// generated.
  OpBuilder b;
  /// The current context.
  MLIRContext *context;
  /// The current module being created.
  ModuleOp module;
  /// The OpenScop object pointer.
  std::unique_ptr<OslScop> &scop;
  /// The symbol table taken from outside.
  OslScopSymbolTable &st;
  /// The CloogOptions that is helpful for debugging purposes.
  CloogOptions *options;

  /// The main function.
  FuncOp f;
};
} // namespace

void Importer::import(clast_stmt *s) {
  for (; s; s = s->next) {
    if (CLAST_STMT_IS_A(s, stmt_root)) {
      processStmt(reinterpret_cast<clast_root *>(s));
    } else if (CLAST_STMT_IS_A(s, stmt_ass)) {
      // TODO: fill this
    } else if (CLAST_STMT_IS_A(s, stmt_user)) {
      processStmt(reinterpret_cast<clast_user_stmt *>(s));
    } else if (CLAST_STMT_IS_A(s, stmt_for)) {
      processStmt(reinterpret_cast<clast_for *>(s));
    } else if (CLAST_STMT_IS_A(s, stmt_guard)) {
      llvm_unreachable("stmt_guard is not implemented yet");
      // TODO: fill this
    } else if (CLAST_STMT_IS_A(s, stmt_block)) {
      llvm_unreachable("stmt_block is not implemented yet");
      // TODO: fill this
    } else {
      assert(false && "Unrecognized statement type.");
    }
  }

  // Post update the function type.
  auto &entryBlock = *f.getBlocks().begin();
  auto funcType = b.getFunctionType(entryBlock.getArgumentTypes(), llvm::None);
  f.setType(funcType);
}

void Importer::initParamSymbols(osl_strings *params) {
  if (params == nullptr)
    return;

  Block &block = f.body().front();
  unsigned numParams = osl_strings_size(params);
  for (unsigned i = 0; i < numParams; i++) {
    llvm::StringRef symbol(params->string[i]);
    if (st.lookup(symbol))
      continue;

    Value arg = block.addArgument(b.getIndexType());
    st.insert(symbol, arg);
  }
}

void Importer::initArraySymbols(osl_arrays *arrays) {
  if (arrays == nullptr)
    return;

  Block &block = f.body().front();
  for (int i = 0; i < arrays->nb_names; i++) {
    llvm::StringRef symbol(arrays->names[i]);
    if (st.lookup(symbol))
      continue;
    llvm::SmallVector<int64_t, 8> shape(scop->getNumDimsOfArray(arrays->id[i]),
                                        -1);
    Value arg = block.addArgument(MemRefType::get(shape, b.getF32Type()));
    st.insert(symbol, arg);
  }
}

/// From the given OpenScop representation, we try to iterate through the symbol
/// table to see if the symbols in <arrays> and scop->parameters already exists.
/// If not, we will create values for them. To create memref values (for
/// <arrays>), we will go through the access relations in all statements, to see
/// if there is any access to the arrays to be created, and if so, what the
/// dimensions of those values. We also need to make sure the number of
/// dimensions are consistent. For parameters, we just create index values for
/// them. All these values will be created as BlockArguments to the provided
/// FuncOp.
/// TODO: initialize based on the provided function interface.
void Importer::initSymbols() {
  initParamSymbols(scop->getParameterNames());
  initArraySymbols(scop->getArrays());
}

void Importer::processStmt(clast_root *rootStmt) {
  llvm::StringRef funSig = scop->getFunctionSignature();
  assert(!funSig.empty() &&
         "You should provide a function signature in the "
         "<strings> tag under scop extension, which should look like foo(A1,"
         "P1), where A1 and P1 are symbols appeared in the OpenScop file.");

  std::string funName;
  llvm::SmallVector<std::string, 8> funArgNames;
  parseFunctionSignature(funSig, funName, funArgNames);

  // The main function to be created has 0 input and output.
  auto funType = b.getFunctionType(llvm::None, llvm::None);
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  f = b.create<FuncOp>(UnknownLoc::get(context), funName, funType);
  auto &entryBlock = *f.addEntryBlock();

  initSymbols();

  // Generate an entry block and implicitly insert a ReturnOp at its end.
  b.setInsertionPoint(&entryBlock, entryBlock.end());
  b.create<mlir::ReturnOp>(UnknownLoc::get(context));

  // For the rest of the body
  b.setInsertionPointToStart(&entryBlock);
}

void Importer::parseFunctionSignature(
    llvm::StringRef body, std::string &calleeName,
    llvm::SmallVectorImpl<std::string> &argNames) {
  unsigned bodyLen = body.size();
  unsigned pos = 0;

  // Read until the left bracket for the function name.
  for (; pos < bodyLen && body[pos] != '('; pos++)
    calleeName.push_back(body[pos]);
  pos++; // Consume the left bracket.

  // Read argument names.
  while (pos < bodyLen) {
    std::string arg;
    for (; pos < bodyLen && body[pos] != ',' && body[pos] != ')'; pos++) {
      if (body[pos] != ' ') // Ignore whitespaces
        arg.push_back(body[pos]);
    }

    argNames.push_back(arg);
    // Consume either ',' or ')'.
    pos++;
  }
}

/// Builds the mapping from the iterator names in a statement to their
/// corresponding names in <scatnames>, based on the matrix provided by the
/// scattering relation.
static void buildIterToScatNameMap(
    llvm::DenseMap<llvm::StringRef, llvm::StringRef> &iterToScatName,
    osl_statement_p stmt, osl_generic_p scatnames) {
  // Get the body from the statement.
  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != nullptr && "The body of the statement should not be NULL.");
  assert(body->expression != nullptr &&
         "The body expression should not be NULL.");
  assert(body->iterators != nullptr &&
         "The body iterators should not be NULL.");

  // Get iterator names.
  unsigned numIterNames = osl_strings_size(body->iterators);
  llvm::SmallVector<llvm::StringRef, 8> iterNames(numIterNames);
  for (unsigned i = 0; i < numIterNames; i++)
    iterNames[i] = body->iterators->string[i];

  // Split the scatnames into a list of strings.
  osl_strings_p scatNamesData =
      reinterpret_cast<osl_scatnames_p>(scatnames->data)->names;
  unsigned numScatNames = osl_strings_size(scatNamesData);

  llvm::SmallVector<llvm::StringRef, 8> scatNames(numScatNames);
  for (unsigned i = 0; i < numScatNames; i++)
    scatNames[i] = scatNamesData->string[i];

  // Get the scattering relation.
  osl_relation_p scats = stmt->scattering;
  assert(scats != nullptr && "scattering in the statement should not be NULL.");
  assert(scats->nb_input_dims == static_cast<int>(iterNames.size()) &&
         "# input dims should equal to # iter names.");
  assert(scats->nb_output_dims <= static_cast<int>(scatNames.size()) &&
         "# output dims should be less than or equal to # scat names.");

  // Build the mapping.
  for (int i = 0; i < scats->nb_output_dims; i++)
    for (int j = 0; j < scats->nb_input_dims; j++)
      if (osl_int_one(osl_util_get_precision(),
                      scats->m[i][j + scats->nb_output_dims + 1]))
        iterToScatName[iterNames[j]] = scatNames[i];
}

/// Create a custom call operation for each user statement. A user statement
/// should be in the format of <stmt-id>`(`<ssa-id>`)`, in which a SSA ID can be
/// a memref, a loop IV, or a symbol parameter (defined as a block argument). We
/// will also generate the declaration of the function to be called, which has
/// an empty body, in order to make the compiler happy.
void Importer::processStmt(clast_user_stmt *userStmt) {
  osl_statement_p stmt = scop->getStatement(userStmt->statement->number - 1);
  assert(stmt != nullptr);

  osl_body_p body = osl_statement_get_body(stmt);
  assert(body != NULL && "The body of the statement should not be NULL.");
  assert(osl_strings_size(body->expression) == 1 &&
         "The statement body should be in one line.");
  assert(body->expression != NULL && "The body expression should not be NULL.");
  assert(body->iterators != NULL && "The body iterators should not be NULL.");

  // Map iterator names in the current statement to the values in <scatnames>.
  osl_generic_p scatnames = scop->getExtension("scatnames");
  assert(scatnames && "There should be a <scatnames> in the scop.");

  llvm::DenseMap<llvm::StringRef, llvm::StringRef> iterToScatName;
  buildIterToScatNameMap(iterToScatName, stmt, scatnames);

  // Parse the statement body.
  llvm::SmallVector<std::string, 8> argNames;
  std::string calleeName;
  // Get the calleeName and the list of argument names (argNames).
  parseFunctionSignature(body->expression->string[0], calleeName, argNames);

  // Cache the current insertion point before changing it for the new callee
  // function.
  auto currBlock = b.getBlock();
  auto currPt = b.getInsertionPoint();

  // Create the callee.
  // First, we create the callee function type. Each arg type is dervied from
  // the type of the values in the symbol table.
  llvm::SmallVector<Value, 8> callerArgs;
  for (const std::string &argName : argNames) {
    bool isIter = iterToScatName.find(argName) != iterToScatName.end();
    Value arg = st.lookup(isIter ? iterToScatName[argName] : argName);
    assert(arg != nullptr &&
           "Each argument should be able to be found in the symbol table.");

    callerArgs.push_back(arg);
  }

  TypeRange calleeArgTypes = ValueRange(callerArgs).getTypes();
  auto calleeType = b.getFunctionType(calleeArgTypes, llvm::None);

  // Then we build the callee.
  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  FuncOp callee =
      b.create<FuncOp>(UnknownLoc::get(context), calleeName, calleeType);
  callee.setVisibility(SymbolTable::Visibility::Private);

  // Create the caller.
  b.setInsertionPoint(currBlock, currPt);

  // Finally create the CallOp.
  b.create<CallOp>(UnknownLoc::get(context), callee, callerArgs);
}

void Importer::getAffineLoopBound(clast_expr *expr,
                                  llvm::SmallVectorImpl<mlir::Value> &operands,
                                  AffineMap &affMap, bool isUpper) {
  AffineExprBuilder builder(context, scop, options);
  SmallVector<AffineExpr, 4> boundExprs;
  // Build the AffineExpr for the loop bound.
  builder.process(expr, boundExprs);

  // The upper bound given by Clast is closed, while affine.for needs an open
  // bound. We go through every boundExpr here.
  if (isUpper)
    for (unsigned i = 0; i < boundExprs.size(); i++)
      boundExprs[i] = boundExprs[i] + b.getAffineConstantExpr(1);

  // ------------------------ Results:
  // Insert dim operands.
  for (auto dimName : builder.dimNames) {
    Value dimValue = st.lookup(dimName);
    assert(dimValue != nullptr &&
           "dimName cannot be found in the symbol table.");
    operands.push_back(dimValue);
  }
  // Insert symbol operands.
  for (auto symbolName : builder.symbolNames) {
    Value symbolValue = st.lookup(symbolName);
    assert(symbolValue != nullptr &&
           "symbolName cannot be found in the symbol table.");
    operands.push_back(symbolValue);
  }
  // Create the AffineMap for loop bound.
  affMap = AffineMap::get(builder.dimNames.size(), builder.symbolNames.size(),
                          boundExprs, context);
}

void Importer::processStmt(clast_for *forStmt) {
  // Get loop bounds.
  AffineMap lbMap, ubMap;
  llvm::SmallVector<mlir::Value, 8> lbOperands, ubOperands;

  // Sanity checks.
  assert((forStmt->LB && forStmt->UB) && "Unbounded loops are not allowed.");
  // TODO: simplify these sanity checks.
  assert(!(forStmt->LB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->LB)->type ==
               clast_red_min) &&
         "If the lower bound is a reduced result, it should not use min for "
         "reduction.");
  assert(!(forStmt->UB->type == clast_expr_red &&
           reinterpret_cast<clast_reduction *>(forStmt->UB)->type ==
               clast_red_max) &&
         "If the lower bound is a reduced result, it should not use max for "
         "reduction.");

  getAffineLoopBound(forStmt->LB, lbOperands, lbMap);
  getAffineLoopBound(forStmt->UB, ubOperands, ubMap, /*isUpper=*/true);

  int64_t stride = 1;
  if (cloog_int_gt_si(forStmt->stride, 1))
    getI64(forStmt->stride, &stride);

  // Create the for operation.
  mlir::AffineForOp forOp = b.create<mlir::AffineForOp>(
      UnknownLoc::get(context), lbOperands, lbMap, ubOperands, ubMap, stride);

  // Update the symbol table by the newly created IV.
  st.insert(forStmt->iterator, forOp.getInductionVar());

  auto &entryBlock = *forOp.getLoopBody().getBlocks().begin();
  // TODO: confirm is there a case that forOp has multiple operands.
  assert(entryBlock.getNumArguments() == 1 &&
         "affine.for should only have one block argument.");
  // Create the loop body
  b.setInsertionPointToStart(&entryBlock);
  import(forStmt->body);
  b.setInsertionPointAfter(forOp);

  // Jumping out of the current region will erase the recently bound iterator.
  st.erase(forStmt->iterator);
}

static std::unique_ptr<OslScop> readOpenScop(llvm::MemoryBufferRef buf) {
  // Read OpenScop by OSL API.
  // TODO: is there a better way to get the FILE pointer from MemoryBufferRef?
  FILE *inputFile = fmemopen(
      reinterpret_cast<void *>(const_cast<char *>(buf.getBufferStart())),
      buf.getBufferSize(), "r");

  auto scop = std::make_unique<OslScop>(osl_scop_read(inputFile));
  fclose(inputFile);

  return scop;
}

Operation *polymer::createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop,
                                             ModuleOp module,
                                             MLIRContext *context) {
  OslScopSymbolTable st;
  return createFuncOpFromOpenScop(std::move(scop), st, module, context);
}

Operation *polymer::createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop,
                                             OslScopSymbolTable &st,
                                             ModuleOp module,
                                             MLIRContext *context) {
  // TODO: turn these C struct into C++ classes.
  CloogState *state = cloog_state_malloc();
  CloogOptions *options = cloog_options_malloc(state);
  options->openscop = 1;

  CloogInput *input = cloog_input_from_osl_scop(options->state, scop->get());
  cloog_options_copy_from_osl_scop(scop->get(), options);

  // Create cloog_program
  CloogProgram *program =
      cloog_program_alloc(input->context, input->ud, options);
  program = cloog_program_generate(program, options);

  // Convert to clast
  clast_stmt *rootStmt = cloog_clast_create(program, options);
  clast_pprint(stdout, rootStmt, 0, options);

  // Process the input.
  Importer deserializer(context, module, scop, st, options);
  deserializer.import(rootStmt);

  // Cannot use cloog_input_free, some pointers don't exist.
  free(input);
  cloog_program_free(program);

  options->scop = NULL; // Prevents freeing the scop object.
  cloog_options_free(options);
  cloog_state_free(state);

  return deserializer.getFunc();
}

OwningModuleRef
polymer::translateOpenScopToModule(std::unique_ptr<OslScop> scop,
                                   MLIRContext *context) {
  context->loadDialect<AffineDialect>();
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));

  if (!createFuncOpFromOpenScop(std::move(scop), module.get(), context))
    return {};

  return module;
}

static OwningModuleRef translateOpenScopToModule(llvm::SourceMgr &sourceMgr,
                                                 MLIRContext *context) {
  llvm::SMDiagnostic err;
  std::unique_ptr<OslScop> scop =
      readOpenScop(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()));

  return translateOpenScopToModule(std::move(scop), context);
}

namespace polymer {

void registerFromOpenScopTranslation() {
  TranslateToMLIRRegistration fromLLVM(
      "import-scop", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return ::translateOpenScopToModule(sourceMgr, context);
      });
}

} // namespace polymer
