add_mlir_library(POLYMEREmitOpenSCoP
  EmitOpenSCoP.cc
  
  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRAffineOps
  )