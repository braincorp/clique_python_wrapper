project(clique_python_wrapper)
set(PROJECT_NAME clique_python_wrapper)
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

find_package(PkgConfig REQUIRED)

## Both Boost.python and Python libs are required.
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs 2.7 REQUIRED)

# Find actual numpy in use (as opposed to system numpy)
execute_process(
COMMAND "python2.7" -c
        "from __future__ import print_function\ntry: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
        OUTPUT_VARIABLE Numpy_INCLUDE_DIRS)

include_directories(
        include
        ${Boost_INCLUDE_DIRS}
        ${Numpy_INCLUDE_DIRS}  # this should come before PYTHON includes because it also contains system numpy
        ${PYTHON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cliquer
        )

##############################################################################
# Define package
##############################################################################

# REMOVE THE FOLLOWING LINES TO COMPILE WITHOUT ROS
catkin_python_setup()
catkin_package()

add_library(max_weighted_clique_module SHARED
  src/cliquer/cliquer.c
  src/cliquer/graph.c
  src/cliquer/reorder.c
  src/max_weighted_clique_module.cpp)

target_link_libraries(max_weighted_clique_module ${Boost_LIBRARIES})

# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(max_weighted_clique_module PROPERTIES
        PREFIX "_wrapper_"
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
        )
