cmake_minimum_required(VERSION 2.8.3)
project(ros_image_processing_mlm)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
)
## System dependencies are found with CMake's conventions
find_package( OpenCV REQUIRED )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ros_lib_imgproc_ml
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs image_transport cv_bridge
  DEPENDS system_lib
)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

## Declare a C++ library
add_library(ros_lib_imgproc_ml src/${PROJECT_NAME}/colorDetection_mlm.cpp )
## Specify libraries to link a library or executable target against
 target_link_libraries(ros_lib_imgproc_ml ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

#############
## Install ##
#############
## Install project namespaced headers
# Install the headers, defining the library interface
install(DIRECTORY include/${PROJECT_NAME}/
		DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
# Install the library source in install/lib folder
install(TARGETS ros_lib_imgproc_ml
		ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
		LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
		RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(ros_image_processing_mlm ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(ros_image_processing_mlm_node src/ros_image_processing_mlm_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(ros_image_processing_mlm_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


