set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(emit-openscop
  emit-openscop.cc
)
llvm_update_compile_flags(emit-openscop)
target_link_libraries(emit-openscop
  PRIVATE
  MLIRTranslation
  MLIRIR
  MLIRStandardOps
  MLIRAffineOps
  MLIRSupport

  POLYMEREmitOpenSCoP
  )