# Build ISL source.

include(ExternalProject)

if (NOT ISL_SOURCE_DIR)
  message(FATAL_ERROR "Should have ISL_SOURCE_DIR properly set.")
endif()

set(ISL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/isl")

# Get isl git commit.
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE ISL_GIT_HASH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                WORKING_DIRECTORY ${ISL_SOURCE_DIR})
message(STATUS "isl git hash: ${ISL_GIT_HASH}")

# TODO: use imath somehow?
string(CONCAT ISL_CONFIGURE_SHELL_SCRIPT
       "#!/usr/bin/env bash\n"
       "\n"
       "${ISL_SOURCE_DIR}/autogen.sh\n"
       "${ISL_SOURCE_DIR}/configure --prefix=${ISL_BINARY_DIR} --with-clang=system\n")
set(ISL_CONFIGURE_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/configure-isl.sh")
file(GENERATE OUTPUT ${ISL_CONFIGURE_COMMAND} CONTENT ${ISL_CONFIGURE_SHELL_SCRIPT})

ExternalProject_Add(
  isl
  PREFIX ${ISL_BINARY_DIR}
  SOURCE_DIR ${ISL_SOURCE_DIR}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env bash ${ISL_CONFIGURE_COMMAND}
  BUILD_COMMAND make -j 4
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
)

add_library(libisl STATIC IMPORTED)
set_target_properties(libisl PROPERTIES
  IMPORTED_LOCATION "${ISL_LIB_DIR}/libisl.a"
  INTERFACE_INCLUDE_DIRECTORIES "${ISL_BINARY_DIR}/include")


