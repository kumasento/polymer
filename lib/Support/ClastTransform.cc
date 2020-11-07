//===- ClastTransform.cc ----------------------------------------*- C++ -*-===//
//
// This file implements clast transforms.
//
//===----------------------------------------------------------------------===//

#include "polymer/Support/ClastTransform.h"

#include "cloog/cloog.h"
#include "osl/osl.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LogicalResult.h"

using namespace llvm;
using namespace mlir;

static std::string getNewIVName(llvm::StringRef fullName) {
  auto ivNamePair = fullName.split("_");
  std::string ivName(ivNamePair.first);

  if (ivNamePair.second.empty()) {
    return ivName + "_2";
  } else {
    std::string idStr(ivNamePair.second);
    unsigned id = std::stoi(idStr);

    return ivName + "_" + std::to_string(id + 1);
  }
}

namespace {

class ClastDupIVTransformer {
public:
  using NameMap = llvm::DenseMap<StringRef, std::string>;

  ClastDupIVTransformer(osl_scop *scop) : scop(scop) {}

  void visit(clast_stmt *stmt);

private:
  LogicalResult visitStmt(clast_stmt *stmt, NameMap &nameMap);
  LogicalResult visitStmt(clast_root *stmt, NameMap &nameMap);
  LogicalResult visitStmt(clast_for *stmt, NameMap &nameMap);
  LogicalResult visitStmt(clast_user_stmt *stmt, NameMap &nameMap);

  LogicalResult visitExpr(clast_expr *expr, NameMap &nameMap);
  LogicalResult visitExpr(clast_name *expr, NameMap &nameMap);
  LogicalResult visitExpr(clast_term *expr, NameMap &nameMap);
  LogicalResult visitExpr(clast_binary *expr, NameMap &nameMap);
  LogicalResult visitExpr(clast_reduction *expr, NameMap &nameMap);

  osl_scop *scop;
};

} // namespace

void ClastDupIVTransformer::visit(clast_stmt *stmt) {
  NameMap nameMap;

  visitStmt(stmt, nameMap);
}

LogicalResult ClastDupIVTransformer::visitExpr(clast_expr *expr,
                                               NameMap &nameMap) {

  switch (expr->type) {
  case clast_expr_name:
    if (failed(visitExpr(reinterpret_cast<clast_name *>(expr), nameMap)))
      return failure();
    break;
  case clast_expr_term:
    if (failed(visitExpr(reinterpret_cast<clast_term *>(expr), nameMap)))
      return failure();
    break;
  case clast_expr_bin:
    if (failed(visitExpr(reinterpret_cast<clast_binary *>(expr), nameMap)))
      return failure();
    break;
  case clast_expr_red:
    if (failed(visitExpr(reinterpret_cast<clast_reduction *>(expr), nameMap)))
      return failure();
    break;
  default:
    assert(false && "Unrecognized clast_expr_type.\n");
    return failure();
  }
  return success();
}

LogicalResult ClastDupIVTransformer::visitExpr(clast_name *expr,
                                               NameMap &nameMap) {
  if (nameMap.find(expr->name) != nameMap.end()) {
    char *newName = (char *)malloc(nameMap[expr->name].size() + 1);
    strcpy(newName, nameMap[expr->name].c_str());
    expr->name = newName;
  }

  return success();
}

LogicalResult ClastDupIVTransformer::visitExpr(clast_term *expr,
                                               NameMap &nameMap) {
  if (!expr->var)
    return success();
  if (failed(visitExpr(expr->var, nameMap)))
    return failure();
  return success();
}

LogicalResult ClastDupIVTransformer::visitExpr(clast_binary *expr,
                                               NameMap &nameMap) {
  if (failed(visitExpr(expr->LHS, nameMap)))
    return failure();

  return success();
}

LogicalResult ClastDupIVTransformer::visitExpr(clast_reduction *expr,
                                               NameMap &nameMap) {
  for (unsigned i = 0; i < expr->n; i++)
    if (failed(visitExpr(expr->elts[i], nameMap)))
      return failure();

  return success();
}

LogicalResult ClastDupIVTransformer::visitStmt(clast_stmt *s,
                                               NameMap &nameMap) {
  for (; s; s = s->next) {
    if (CLAST_STMT_IS_A(s, stmt_root)) {
      if (failed(visitStmt(reinterpret_cast<clast_root *>(s), nameMap)))
        return failure();
    } else if (CLAST_STMT_IS_A(s, stmt_ass)) {
      // TODO: fill this
    } else if (CLAST_STMT_IS_A(s, stmt_user)) {
      if (failed(visitStmt(reinterpret_cast<clast_user_stmt *>(s), nameMap)))
        return failure();
    } else if (CLAST_STMT_IS_A(s, stmt_for)) {
      if (failed(visitStmt(reinterpret_cast<clast_for *>(s), nameMap)))
        return failure();
    } else {
      llvm::errs() << "clast_stmt type not supported\n";
      return failure();
    }
  }

  return success();
}

LogicalResult ClastDupIVTransformer::visitStmt(clast_root *s,
                                               NameMap &nameMap) {
  return success();
}

LogicalResult ClastDupIVTransformer::visitStmt(clast_for *s, NameMap &nameMap) {
  // Update loop bounds based on the existing mapping.
  visitExpr(s->LB, nameMap);
  visitExpr(s->UB, nameMap);

  // If the current iterator appears somewhere else, we should create a mapping
  // for it. There are two scenarios:
  // 1. If the iterator doesn't appear to be in the map, we just insert a
  // mapping that maps it to itself.
  // 2. Otherwise (the iterator is in the map), we update the mapping based on
  // the value, e.g., if the value is `i_2`, we update it to `i_3`.

  // Use to recover the previous mapping.
  std::string oldName;

  if (nameMap.find(s->iterator) == nameMap.end()) {
    oldName = std::string(s->iterator);
    nameMap[s->iterator] = oldName;
  } else {
    oldName = nameMap[s->iterator];
    nameMap[s->iterator] = getNewIVName(s->iterator);
  }

  // Make the effect of changing the loop iterator.
  char *newIterName = (char *)malloc(nameMap[s->iterator].size() + 1);
  strcpy(newIterName, nameMap[s->iterator].c_str());
  s->iterator = newIterName;

  visitStmt(s->body, nameMap);

  nameMap[s->iterator] = oldName;

  return success();
}

LogicalResult ClastDupIVTransformer::visitStmt(clast_user_stmt *s,
                                               NameMap &nameMap) {
  return success();
}

void polymer::transformClastDupIV(clast_stmt *stmt, osl_scop *scop,
                                  CloogOptions *options) {
  ClastDupIVTransformer(scop).visit(stmt);
}
