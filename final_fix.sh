#!/bin/bash

echo "=== 最终诊断和修复 ==="

echo "1. 检查 ROS 库文件..."
for lib in roscpp rosconsole roscpp_serialization rostime xmlrpcpp cpp_common; do
    if [ -f "/opt/ros/noetic/lib/lib${lib}.so" ]; then
        echo "✓ lib${lib}.so 存在"
    else
        echo "✗ lib${lib}.so 不存在"
    fi
done

echo "2. 检查环境变量..."
echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH" | tr ':' '\n'
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | tr ':' '\n'

echo "3. 重新配置工作空间..."
rm -rf build devel
catkin_make --force-cmake

echo "4. 编译 unitree_guide..."
catkin_make --pkg unitree_guide -j1
