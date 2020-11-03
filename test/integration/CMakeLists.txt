# Create integration_utils library that contains essential utilities.
add_library(integration_utils
  "${CMAKE_CURRENT_SOURCE_DIR}/utils/polybench.c"
)
target_compile_definitions(
  integration_utils
  PUBLIC
  POLYBENCH_TIME
)

# add_integration_test creates a CMake compilation target that follows:
# - There should be a source file that corresponds to ${NAME}.c
# - It will be linked to integration_utils
# - There should be 2 MLIR source files: ${NAME}.mlir, ${NAME}_opt.mlir
function(add_integration_test NAME)
  set(TARGET "integration_${NAME}")
  add_executable("${TARGET}" "${NAME}.c")

  target_link_libraries(
    "${TARGET}"
    PRIVATE
    integration_utils
  )
  target_include_directories(
    "${TARGET}"
    PRIVATE
    "${CMAKE_SOURCE_DIR}/test/integration/utils"
  )
  target_compile_options(
    "${TARGET}"
    PRIVATE
    -O3
  )
endfunction()

add_subdirectory(matmul)
