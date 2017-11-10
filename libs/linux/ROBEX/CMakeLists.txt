cmake_minimum_required(VERSION 2.6)
PROJECT(RobustStrip)
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR
          "ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)
ADD_EXECUTABLE(ROBEX RobustStrip.cxx aux_methods.cxx   loadModel.cpp graph.cpp maxflow.cpp maxflow_main.cpp ) 
TARGET_LINK_LIBRARIES(ROBEX ${ITK_LIBRARIES})


SET_TARGET_PROPERTIES(
  ROBEX
  PROPERTIES
  LINKER_LANGUAGE C 
  )
