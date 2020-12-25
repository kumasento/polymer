add_mlir_library(PolymerSupport
  OslScop.cc
  OslScopStmtOpSet.cc
  OslSymbolTable.cc
  ScatUtils.cc

  ADDITIONAL_HEADER_DIRS
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Support

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLoopAnalysis

  libplutoisl
  libplutoosl
  libplutopip
  libplutocloog
  libplutocandl
  libpluto
)

target_link_libraries(PolymerSupport PRIVATE ${GMP_LIBRARY})
