#-------------------------------------------------------------------------------
# SuiteSparse/GraphBLAS/cmake_modules/GraphBLASConfig.cmake
#-------------------------------------------------------------------------------

# The following copyright and license applies to just this file only, not to
# the library itself:
# GraphBLASConfig.cmake, Copyright (c) 2023-2025, FIXME
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# Finds the GraphBLAS_CUDA include file and compiled library.
# The following targets are defined:
#   SuiteSparse::GRAPHBLAS_CUDA           - for the shared library (if available)
#   SuiteSparse::GRAPHBLAS_CUDA_static    - for the static library (if available)

# For backward compatibility the following variables are set:

# GRAPHBLAS_CUDA_INCLUDE_DIR - where to find GraphBLAS.h, etc.
# GRAPHBLAS_CUDA_LIBRARY     - dynamic GraphBLAS library
# GRAPHBLAS_CUDA_STATIC      - static GraphBLAS library
# GRAPHBLAS_CUDA_LIBRARIES   - libraries when using GraphBLAS
# GRAPHBLAS_CUDA_FOUND       - true if GraphBLAS found

# Set ``CMAKE_MODULE_PATH`` to the parent folder where this module file is
# installed.

#-------------------------------------------------------------------------------


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was GraphBLAS_CUDAConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set ( GRAPHBLAS_CUDA_DATE "Nov 1, 2025" )
set ( GRAPHBLAS_CUDA_VERSION_MAJOR 10 )
set ( GRAPHBLAS_CUDA_VERSION_MINOR 2 )
set ( GRAPHBLAS_CUDA_VERSION_PATCH 0 )
set ( GRAPHBLAS_CUDA_VERSION "10.2.0" )

# Check for dependent targets
include ( CMakeFindDependencyMacro )
set ( _dependencies_found ON )

# Look for NVIDIA CUDA toolkit
if ( NOT CUDAToolkit_FOUND )
    find_dependency ( CUDAToolkit 13 )
    if ( NOT CUDAToolkit_FOUND )
        set ( _dependencies_found OFF )
    endif ( )
endif ( )

if ( NOT _dependencies_found )
    set ( GraphBLAS_CUDA_FOUND OFF )
    return ( )
endif ( )

# Import target
include ( ${CMAKE_CURRENT_LIST_DIR}/GraphBLAS_CUDATargets.cmake )

# The following is only for backward compatibility with FindGraphBLAS_CUDA.

set ( _target_shared SuiteSparse::GraphBLAS_CUDA )
set ( _target_static SuiteSparse::GraphBLAS_CUDA_static )
set ( _var_prefix "GRAPHBLAS_CUDA" )

get_target_property ( ${_var_prefix}_INCLUDE_DIR ${_target_shared} INTERFACE_INCLUDE_DIRECTORIES )
if ( ${_var_prefix}_INCLUDE_DIR )
    # First item in SuiteSparse targets contains the "main" header directory.
    list ( GET ${_var_prefix}_INCLUDE_DIR 0 ${_var_prefix}_INCLUDE_DIR )
endif ( )
get_target_property ( ${_var_prefix}_LIBRARY ${_target_shared} IMPORTED_IMPLIB )
if ( NOT ${_var_prefix}_LIBRARY )
    get_target_property ( _library_chk ${_target_shared} IMPORTED_LOCATION )
    if ( EXISTS ${_library_chk} )
        set ( ${_var_prefix}_LIBRARY ${_library_chk} )
    endif ( )
endif ( )
if ( TARGET ${_target_static} )
    get_target_property ( ${_var_prefix}_STATIC ${_target_static} IMPORTED_LOCATION )
endif ( )

# Check for most common build types
set ( _config_types "Debug" "Release" "RelWithDebInfo" "MinSizeRel" "None" )

get_property ( _isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG )
if ( _isMultiConfig )
    # For multi-configuration generators (e.g., Visual Studio), prefer those
    # configurations.
    list ( PREPEND _config_types ${CMAKE_CONFIGURATION_TYPES} )
else ( )
    # For single-configuration generators, prefer the current configuration.
    list ( PREPEND _config_types ${CMAKE_BUILD_TYPE} )
endif ( )

list ( REMOVE_DUPLICATES _config_types )

foreach ( _config ${_config_types} )
    string ( TOUPPER ${_config} _uc_config )
    if ( NOT ${_var_prefix}_LIBRARY )
        get_target_property ( _library_chk ${_target_shared}
            IMPORTED_IMPLIB_${_uc_config} )
        if ( EXISTS ${_library_chk} )
            set ( ${_var_prefix}_LIBRARY ${_library_chk} )
        endif ( )
    endif ( )
    if ( NOT ${_var_prefix}_LIBRARY )
        get_target_property ( _library_chk ${_target_shared}
            IMPORTED_LOCATION_${_uc_config} )
        if ( EXISTS ${_library_chk} )
            set ( ${_var_prefix}_LIBRARY ${_library_chk} )
        endif ( )
    endif ( )
    if ( TARGET ${_target_static} AND NOT ${_var_prefix}_STATIC )
        get_target_property ( _library_chk ${_target_static}
            IMPORTED_LOCATION_${_uc_config} )
        if ( EXISTS ${_library_chk} )
            set ( ${_var_prefix}_STATIC ${_library_chk} )
        endif ( )
    endif ( )
endforeach ( )

set ( GRAPHBLAS_CUDA_LIBRARIES ${GRAPHBLAS_CUDA_LIBRARY} )

macro ( suitesparse_check_exist _var _files )
  # ignore generator expressions
  string ( GENEX_STRIP "${_files}" _files2 )

  foreach ( _file ${_files2} )
    if ( NOT EXISTS "${_file}" )
      message ( FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist!" )
    endif ( )
  endforeach ()
endmacro ( )

suitesparse_check_exist ( GRAPHBLAS_CUDA_INCLUDE_DIR ${GRAPHBLAS_CUDA_INCLUDE_DIR} )
suitesparse_check_exist ( GRAPHBLAS_CUDA_LIBRARY ${GRAPHBLAS_CUDA_LIBRARY} )

message ( STATUS "GraphBLAS_CUDA version: ${GRAPHBLAS_CUDA_VERSION}" )
message ( STATUS "GraphBLAS_CUDA include: ${GRAPHBLAS_CUDA_INCLUDE_DIR}" )
message ( STATUS "GraphBLAS_CUDA library: ${GRAPHBLAS_CUDA_LIBRARY}" )
message ( STATUS "GraphBLAS_CUDA static:  ${GRAPHBLAS_CUDA_STATIC}" )
