cmake_minimum_required(VERSION 3.14)
project(unitree_guide)

# if(CMAKE_COMPILER_IS_GNUCXX)
#     add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
#     message(STATUS "Add compile option -D_GLIBCXX_USE_CXX11_ABI=1")
# endif()

set(ROBOT_TYPE Go1)         # The type of robot, support A1 Go1 Go2
set(PLATFORM amd64)         # The platform to compile, support amd64 and arm64

set(CATKIN_MAKE ON)        # Use CATKIN_MAKE or not, ON or OFF
set(SIMULATION ON)         # Use Gazebo or not, ON or OFF
set(REAL_ROBOT OFF)          # Link real robot or not, ON or OFF
set(DEBUG OFF)              # Use debug functions or not, ON or OFF
set(MOVE_BASE OFF)               # Need move_base or not, ON or OFF 

set(use_Torch ON)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)


if(use_Torch)
    set(Torch_DIR /home/zzf/install_source/Libtorch/libtorch-cxx11-abi-shared-with-deps-1.10.0+cpu/libtorch/share/cmake/Torch)  # set libtorch path
    find_package(Torch REQUIRED)
    
    set(BODY_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/model/body.jit")
    set(ADAPT_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/model/adapt.jit")
    add_definitions(-DBODY_MODEL_PATH="${BODY_MODEL_PATH}")
    add_definitions(-DADAPT_MODEL_PATH="${ADAPT_MODEL_PATH}")
endif()

if(NOT DEFINED ROBOT_TYPE)
    message(FATAL_ERROR "[CMake ERROR] Have not defined ROBOT_TYPE")
endif()
if(NOT DEFINED PLATFORM)
    message(FATAL_ERROR "[CMake ERROR] Have not defined PLATFORM")
endif()

if(${ROBOT_TYPE} STREQUAL "A1")
    add_definitions(-DROBOT_TYPE_A1)
elseif(${ROBOT_TYPE} STREQUAL "Go1")
    add_definitions(-DROBOT_TYPE_Go1)
elseif(${ROBOT_TYPE} STREQUAL "Go2")
    add_definitions(-DROBOT_TYPE_Go2)
else()
    message(FATAL_ERROR "[CMake ERROR] The ROBOT_TYPE is error")
endif()

if(((SIMULATION) AND (REAL_ROBOT)) OR ((NOT SIMULATION) AND (NOT REAL_ROBOT)))
    message(FATAL_ERROR "[CMake ERROR] The SIMULATION and REAL_ROBOT can only be one ON one OFF")
endif()

if(SIMULATION OR MOVE_BASE)
    add_definitions(-DRUN_ROS)
    set(CATKIN_MAKE ON)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost)

if(CATKIN_MAKE)
    add_definitions(-DCOMPILE_WITH_ROS)
    if(MOVE_BASE)
        add_definitions(-DCOMPILE_WITH_MOVE_BASE)
    endif()
endif()

if(DEBUG)
    add_definitions(-DCOMPILE_DEBUG)
    find_package(Python2 COMPONENTS Interpreter Development NumPy)
endif()

if(CATKIN_MAKE)
    if(SIMULATION)
        add_definitions(-DCOMPILE_WITH_SIMULATION)
        find_package(catkin REQUIRED COMPONENTS
            controller_manager
            joint_state_controller
            gazebo_ros
            # gazebo
        )
    endif()

    find_package(catkin REQUIRED COMPONENTS
        genmsg
        robot_state_publisher
        roscpp
        std_msgs
        tf
        geometry_msgs
        unitree_legged_msgs
        rosconsole 
        rospy       
        roslib
    )
    catkin_package(
        CATKIN_DEPENDS 
        unitree_legged_msgs 
    )
endif()

include_directories(
    include
    ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${GAZEBO_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS} -O3")

if(REAL_ROBOT)
    add_definitions(-DCOMPILE_WITH_REAL_ROBOT)
    if(${ROBOT_TYPE} STREQUAL "A1")
        include_directories(
            library/unitree_legged_sdk_3.2/include
        )
        link_directories(
            library/unitree_legged_sdk_3.2/lib
        )
    elseif(${ROBOT_TYPE} STREQUAL "Go1")
        include_directories(
            library/unitree_legged_sdk-3.8.0/include
        )
        if(${PLATFORM} STREQUAL "amd64")
            link_directories(
                library/unitree_legged_sdk-3.8.0/lib/cpp/amd64
            )
        elseif(${PLATFORM} STREQUAL "arm64")
            link_directories(
                library/unitree_legged_sdk-3.8.0/lib/cpp/arm64
            )
        endif()
    elseif(${ROBOT_TYPE} STREQUAL "Go2")
        include_directories(
            /usr/local/include/ddscxx
            /usr/local/include/iceoryx/v2.0.2
            /usr/local/include/unitree
            ./library/unitree_sdk2/include
        )
        link_directories(/usr/local/lib)
    endif()
endif()

link_directories(
    ${GAZEBO_LIBRARY_DIRS}
)

file(GLOB_RECURSE SRC_LIST
    "src/*/*.cpp"
    "src/*/*.cc"
)

add_executable(junior_ctrl src/main.cpp ${SRC_LIST})
# if(use_Torch)
#     target_link_libraries(junior_ctrl ${TORCH_LIBRARIES})
# endif()
if(CATKIN_MAKE)

    # find_library(ROS_LIB_ROSCPP roscpp)
    # find_library(ROS_LIB_ROSCOMMON roscommon)
    # find_library(ROS_LIB_ROSCONSOLE rosconsole)
    # find_library(ROS_LIB_ROSCPP_SERIALIZATION roscpp_serialization)
    # find_library(ROS_LIB_ROSLOG roslog)
    
    # message(STATUS "Found roscpp: ${ROS_LIB_ROSCPP}")
    # message(STATUS "Found roscommon: ${ROS_LIB_ROSCOMMON}")
    # message(STATUS "Found rosconsole: ${ROS_LIB_ROSCONSOLE}")

    link_directories(/opt/ros/noetic/lib)

    target_link_libraries(junior_ctrl
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES}
        -pthread 
        lcm
        /opt/ros/noetic/lib/libroscpp.so
        /opt/ros/noetic/lib/librosconsole.so
        /opt/ros/noetic/lib/librosconsole_log4cxx.so
        /opt/ros/noetic/lib/librosconsole_backend_interface.so
        /opt/ros/noetic/lib/libroscpp_serialization.so
        /opt/ros/noetic/lib/librostime.so
        /opt/ros/noetic/lib/libcpp_common.so
    )
    add_dependencies(junior_ctrl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
    set_property(TARGET junior_ctrl PROPERTY CXX_STANDARD 14)

    target_compile_options(junior_ctrl PRIVATE ${Torch_CXX_FLAGS})
endif()
if(DEBUG)
    target_include_directories(junior_ctrl PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
    target_link_libraries(junior_ctrl Python2::Python Python2::NumPy)
endif()
if(REAL_ROBOT)
    if(${ROBOT_TYPE} STREQUAL "A1")
        if(${PLATFORM} STREQUAL "amd64")
            target_link_libraries(junior_ctrl libunitree_legged_sdk_amd64.so)
        elseif(${PLATFORM} STREQUAL "arm64")
            target_link_libraries(junior_ctrl libunitree_legged_sdk_arm64.so)
        endif()
    elseif(${ROBOT_TYPE} STREQUAL "Go1")
            target_link_libraries(junior_ctrl libunitree_legged_sdk.a)
    elseif(${ROBOT_TYPE} STREQUAL "Go2")
            target_link_libraries(junior_ctrl unitree_sdk2 ddsc ddscxx rt pthread)
    endif()
endif()
# target_link_libraries(junior_ctrl -pthread lcm)

if(NOT CATKIN_MAKE)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
endif()