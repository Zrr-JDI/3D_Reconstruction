cmake_minimum_required(VERSION 3.20)

project(3D_RECONSTRUCTION)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件和头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# 手动指定 OpenCV 的 build 目录（不要指向 lib）
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ThirdPart/OpenCV/build)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 可执行文件
add_executable(app main.cpp ${SRC} "include/CameraCalibrator.h")
target_link_libraries(app ${OpenCV_LIBS})

# 获取 OpenCV 运行时 DLL 所在路径
if(OpenCV_DLL_DIR)
    set(OPENCV_DLL_DIR ${OpenCV_DLL_DIR})
elseif(EXISTS "${OpenCV_DIR}/x64/vc16/bin")
    set(OPENCV_DLL_DIR "${OpenCV_DIR}/x64/vc16/bin")
elseif(EXISTS "${OpenCV_DIR}/bin/Debug")
    set(OPENCV_DLL_DIR "${OpenCV_DIR}/bin/Debug")
elseif(EXISTS "${OpenCV_DIR}/bin/Release")
    set(OPENCV_DLL_DIR "${OpenCV_DIR}/bin/Release")
else()
    message(WARNING "⚠️ 未找到 OpenCV DLL 目录，请检查 OpenCV 安装路径")
endif()

# 自动拷贝所有 DLL 到输出目录（确保可执行文件能直接运行）
if(EXISTS "${OPENCV_DLL_DIR}")
    file(GLOB OPENCV_DLLS "${OPENCV_DLL_DIR}/*.dll")
    add_custom_command(TARGET app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DLLS}
        $<TARGET_FILE_DIR:app>)
endif()
