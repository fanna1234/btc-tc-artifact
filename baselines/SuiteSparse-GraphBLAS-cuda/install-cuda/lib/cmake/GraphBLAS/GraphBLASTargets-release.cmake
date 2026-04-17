#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::RMM_wrap" for configuration "Release"
set_property(TARGET SuiteSparse::RMM_wrap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::RMM_wrap PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librmm_wrap.so.10.2.0"
  IMPORTED_SONAME_RELEASE "librmm_wrap.so.10"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::RMM_wrap )
list(APPEND _cmake_import_check_files_for_SuiteSparse::RMM_wrap "${_IMPORT_PREFIX}/lib/librmm_wrap.so.10.2.0" )

# Import target "SuiteSparse::GraphBLAS" for configuration "Release"
set_property(TARGET SuiteSparse::GraphBLAS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::GraphBLAS PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::GraphBLAS_CUDA;SuiteSparse::RMM_wrap"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgraphblas.so.10.2.0"
  IMPORTED_SONAME_RELEASE "libgraphblas.so.10"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::GraphBLAS )
list(APPEND _cmake_import_check_files_for_SuiteSparse::GraphBLAS "${_IMPORT_PREFIX}/lib/libgraphblas.so.10.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
