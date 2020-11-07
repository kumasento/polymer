//===- ClastTransform.h -----------------------------------------*- C++ -*-===//
//
// This file declares clast transforms.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_CLASTTRANSFORM_H
#define POLYMER_SUPPORT_CLASTTRANSFORM_H

struct clast_stmt;
struct cloogoptions;
struct osl_scop;

namespace polymer {

/// In-place rename the loop IVs in the clast AST that share the same name.
void transformClastDupIV(clast_stmt *stmt, osl_scop *scop,
                         cloogoptions *options);

} // namespace polymer

#endif
