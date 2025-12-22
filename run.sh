#!/bin/bash
# 使用wmctrl调整窗口位置

# 获取屏幕尺寸
screen_width=$(xrandr | grep '*' | head -1 | awk '{print $1}' | cut -d 'x' -f1)
screen_height=$(xrandr | grep '*' | head -1 | awk '{print $1}' | cut -d 'x' -f2)

# 计算窗口大小和位置
term_width=$((screen_width / 2 - 20))
term_height=$((screen_height / 2 - 20))


# 打开4个窗口
gnome-terminal --title="Gazebo Simulation" -- bash -c "echo 'Top-left';source ./devel/setup.bash; roslaunch unitree_guide gazeboSim.launch " & 
sleep 0.5

gnome-terminal --title="Junior controller" -- bash -c "echo 'Top-right';source ./devel/setup.bash; rosrun unitree_guide junior_ctrl " &
sleep 0.5

gnome-terminal --title="action" -- bash -c "echo 'Bottom-left';source ./devel/setup.bash; rosrun rl action.py" &
sleep 0.5

gnome-terminal --title="depth_process" -- bash -c "echo 'Bottom-right';source ./devel/setup.bash; rosrun rl depth_process.py" &
sleep 0.5

# 使用wmctrl调整窗口位置和大小
wmctrl -r "Gazebo Simulation" -e 0,0,0,$term_width,$term_height
wmctrl -r "Junior controller" -e 0,$((term_width + 200)),0,$term_width,$term_height
wmctrl -r "action" -e 0,0,$((term_height + 200)),$term_width,$term_height
wmctrl -r "depth_process" -e 0,$((term_width + 200)),$((term_height + 200)),$term_width,$term_height
