add_mlir_library(PolymerSupportTestPasses
  TestScatTree.cc
  TestOslScopBuilder.cc

  ADDITIONAL_HEADER_DIRS
  "${POLYMER_MAIN_INCLUDE_DIR}/polymer/Support"

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  PolymerSupport
)
