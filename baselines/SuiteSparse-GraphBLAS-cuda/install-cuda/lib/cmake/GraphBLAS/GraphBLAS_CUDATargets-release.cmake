#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::GraphBLAS_CUDA" for configuration "Release"
set_property(TARGET SuiteSparse::GraphBLAS_CUDA APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::GraphBLAS_CUDA PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgraphblascuda.so.10.2.0"
  IMPORTED_SONAME_RELEASE "libgraphblascuda.so.10"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::GraphBLAS_CUDA )
list(APPEND _cmake_import_check_files_for_SuiteSparse::GraphBLAS_CUDA "${_IMPORT_PREFIX}/lib/libgraphblascuda.so.10.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
