cmake_minimum_required (VERSION 2.6)
set (PROJECT_NAME gimx-calibration)
project (${PROJECT_NAME})

# Compile flags
SET (CMAKE_C_FLAGS                "-Wall -std=c11 -Wextra")
SET (CMAKE_C_FLAGS_DEBUG          "${CMAKE_CFLAGS} -g")
SET (CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_CFLAGS} -Os -DNDEBUG")
SET (CMAKE_C_FLAGS_RELEASE        "${CMAKE_CFLAGS} -O3 -DNDEBUG")
SET (CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CFLAGS} -O2 -g")

SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11 -Wextra -fPIE") 
SET (CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXXFLAGS} -g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXXFLAGS} -Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXXFLAGS} -O2 -g")
SET (CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXXFLAGS} -O3 -DNDEBUG")

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")
include (cotire)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets REQUIRED)
find_package (Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package (OpenCV REQUIRED)
find_package (cpp-matplotlib REQUIRED)

add_definitions(${Qt5Widgets_DEFINITIONS})

## Set optional features.  This will show up as a preprocessor variable
option (USE_GLIBCXX_DEBUG 
  "Use the _GLIBCXX_DEBUG macro for nicer debugging and error checking of the c++ standard library.  Warning, it breaks ABI compatibility so don't pass container instantiations between translation units that don't have the same debug mode."
  OFF)
## USE_MY_LIBRARY in source.
#option (USE_MY_LIBRARY
#        "Use the provided library" ON)

SET (CMAKE_C_COMPILER             "clang")
SET (CMAKE_CXX_COMPILER             "clang++")

## Support for Clang's CompilationDatabase system
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)

## Compile and create a library.  STATIC is default unless BUILD_SHARED_LIBS
## is on.
#add_library (LibraryName src/sources.cc)

#if (USE_MY_LIBRARY)
  ## Search for include files here as well
  #include_directories ("{PROJECT_SOURCE_DIR}/some_sub_path")

  ## Run Cmake also in this dir
  #add_subdirectory (some_sub_path)

  #set (EXTRA_LIBS ${EXTRA_LIBS} LibraryName)
#endif (USE_MY_LIBRARY)

set (LIBS ${cpp-matplotlib_LIBS} ${OpenCV_LIBS})

add_executable (${PROJECT_NAME} src/main.cc src/headers_pch.h)
target_link_libraries (${PROJECT_NAME} ${EXTRA_LIBS} ${LIBS} flycapture)
set_target_properties (
  ${PROJECT_NAME} PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "src/headers_pch.h")
#cotire (${PROJECT_NAME})

add_executable (control src/control.cc)
qt5_use_modules(control Widgets)
target_link_libraries (${PROJECT_NAME} 
  ${EXTRA_LIBS} ${LIBS} 
  ${Qt5Widgets_LIBRARIES})

if (USE_GLIBCXX_DEBUG)
  # Prefer using G++ for debug mode because the GLIBCXX_DEBUG flag for pretty
  # printing collections in GDB doesn't seem to work with clang.
  string(TOUPPER ${CMAKE_BUILD_TYPE} DEBUG_MODE)
  if (DEBUG_MODE MATCHES "DEBUG")
    set (CMAKE_CXX_COMPILER "g++")
    set (CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXXFLAGS} -g -D_GLIBCXX_DEBUG")
  elseif (DEBUG_MODE MATCHES "RELWITHDEBINFO")
    set (CMAKE_CXX_COMPILER "g++")
    set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXXFLAGS} -O2 -g -D_GLIBCXX_DEBUG")
  endif (DEBUG_MODE MATCHES "DEBUG")
endif (USE_GLIBCXX_DEBUG)


