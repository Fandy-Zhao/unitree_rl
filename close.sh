#!/bin/bash
# 关闭所有ROS窗口脚本

echo "正在关闭所有ROS节点窗口..."

# 关闭所有相关窗口
wmctrl -c "Gazebo Simulation" 2>/dev/null || true
wmctrl -c "Junior controller" 2>/dev/null || true
wmctrl -c "action" 2>/dev/null || true
wmctrl -c "depth_process" 2>/dev/null || true

# 停止ROS节点
rosnode kill -a 2>/dev/null || true
killall -9 roslaunch 2>/dev/null || true
killall -9 roscore 2>/dev/null || true

# 等待并检查是否还有窗口未关闭
sleep 2
for title in "Gazebo Simulation" "Junior controller" "action" "depth_process"; do
    window_id=$(xdotool search --name "$title" 2>/dev/null | head -1)
    if [ -n "$window_id" ]; then
        echo "强制关闭: $title"
        xdotool windowkill $window_id 2>/dev/null || true
        echo "关闭窗口: $title"
        wmctrl -c "$title" 2>/dev/null
    fi
done

echo "所有窗口已关闭！"