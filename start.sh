#!/bin/bash

gnome-terminal -- bash -c "source devel/setup.bash && roslaunch unitree_guide gazeboSim.launch"

gnome-terminal -- bash -c "source devel/setup.bash && rosrun unitree_guide junior_ctrl"