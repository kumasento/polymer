add_mlir_library(PolymerSupportTestPasses
  TestScatTree.cc

  ADDITIONAL_HEADER_DIRS
  "${POLYMER_INCLUDE_DIR}/polymer/Support"

  LINK_COMPONENTS
  Core

  LINK_LIBS 
  PolymerSupport
)
