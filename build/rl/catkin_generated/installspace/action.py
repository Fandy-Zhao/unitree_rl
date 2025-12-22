#!/usr/bin/env python3
import rospy
import numpy as np
import torch
from torch import nn
import json
import os
import os.path as osp
from collections import OrderedDict
import time
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from rl.unitree_ros_real import get_euler_xyz, UnitreeRosReal

from unitree_legged_msgs.msg import LowCmd, LowState, MotorCmd, MotorState
from sensor_msgs.msg import Image
from std_msgs.msg import Header
# import cv2
from cv_bridge import CvBridge

# from sport_api_constants import *

class Action(UnitreeRosReal):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_class_name= "Go2", **kwargs)
        self.global_counter = 0
        self.visual_update_interval = 5

        self.actions_sim = torch.from_numpy(np.load(r'/home/zzf/RL/unitree_rl/src/rl/Action_sim_335-11_flat.npy')).to(self.model_device)

        self.sim_ite = 3
 
        self.use_stand_policy = False
        self.use_parkour_policy = True
        self.use_sport_mode = False
        self.duration = 0.01
        
        print("Action init done")
        
        self.start_ros_handlers()
        

    def float32multiarray_to_tensor(self, msg):
        import numpy as np
        import torch
        # 检查数据是否为空
        if len(msg.data) == 0:
            print("[Warning] Float32MultiArray data is empty!")
            return torch.zeros(1, 58, 87, device=self.model_device)
    
        if isinstance(msg, torch.Tensor):
            # 如果是张量，直接移动到指定设备
            return msg.to(self.model_device)
        elif hasattr(msg, 'data'):
            # 如果是ROS消息，转换为numpy，再转为张量，然后放到指定设备
            arr = np.array(msg.data, dtype=np.float32)
            return torch.from_numpy(arr).to(self.model_device)
        else:
            raise TypeError("Unsupported type for float32multiarray_to_tensor: {}".format(type(msg)))
    
    def warm_up(self):
        """Warm up the policy with initial iterations"""
        for _ in range(2):
            start_time = time.time()
            
            proprio = self.get_proprio()
            get_pro_time = time.time()
            
            proprio_history = self._get_history_proprio()
            get_hist_pro_time = time.time()
            
            depth_image = self._get_depth_image()
            depth_image = self.float32multiarray_to_tensor(depth_image)
            if self.depth_encode:
                self.depth_latent_yaw = self.depth_encode(depth_image, proprio)
                
            get_obs_time = time.time()
            
            if self.turn_obs:
                obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, 
                                  self.n_proprio, self.n_depth_latent, self.n_hist_len)
                                  
            turn_obs_time = time.time()
            
            if self.policy:
                action = self.policy(obs)
                
            policy_time = time.time()
            
            rospy.loginfo(f"Warm up: "
                         f"get proprio time: {get_pro_time - start_time:.5f}, "
                         f"get hist pro time: {get_hist_pro_time - get_pro_time:.5f}, "
                         f"get_depth time: {get_obs_time - get_hist_pro_time:.5f}, "
                         f"get obs time: {get_obs_time - start_time:.5f}, "
                         f"turn_obs_time: {turn_obs_time - get_obs_time:.5f}, "
                         f"policy_time: {policy_time - turn_obs_time:.5f}, "
                         f"total time: {policy_time - start_time:.5f}")
                         
    def main_loop(self, event=None):
        """Main control loop"""
        if self.use_sport_mode:
            # Handle sport mode transitions based on joystick input
            # Note: You'll need to implement joystick handling based on your actual joystick messages
            pass
            
        if self.use_stand_policy:
            stand_action = self.get_stand_action()
            self.send_stand_action(stand_action)
            
        if self.use_parkour_policy:
            self.use_stand_policy = False
            self.use_sport_mode = False
            
            #收集当前状态observation
            start_time = time.time()
            
            proprio = self.get_proprio()
            get_pro_time = time.time()
            
            proprio_history = self._get_history_proprio()
            get_hist_pro_time = time.time()
            
            if self.global_counter % self.visual_update_interval == 0:
                depth_image = self._get_depth_image()
                if self.global_counter == 0:
                    self.last_depth_image = depth_image
                if self.depth_encode:
                    self.depth_latent_yaw = self.depth_encode(self.last_depth_image, proprio)
                self.last_depth_image = depth_image
                
            get_obs_time = time.time()
            
            if self.turn_obs:
                obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history,
                                  self.n_proprio, self.n_depth_latent, self.n_hist_len)
                                  
            turn_obs_time = time.time()
            
            if self.policy:
                action = self.policy(obs)
                
            policy_time = time.time()
            
            # Uncomment to use simulation actions instead of policy
            # action = self.actions_sim[self.sim_ite, :]
            # self.sim_ite += 1
            
            self.send_action(action)
            # rospy.logdebug(f'Action: {action}')
            
            publish_time = time.time()
            
            # rospy.loginfo(f"Loop timings: "
            #              f"get proprio: {get_pro_time - start_time:.5f}, "
            #              f"get hist pro: {get_hist_pro_time - get_pro_time:.5f}, "
            #              f"get_depth: {get_obs_time - get_hist_pro_time:.5f}, "
            #              f"get obs: {get_obs_time - start_time:.5f}, "
            #              f"turn_obs: {turn_obs_time - get_obs_time:.5f}, "
            #              f"policy: {policy_time - turn_obs_time:.5f}, "
            #              f"total: {publish_time - start_time:.5f}")
                         
            self.global_counter += 1
            
    def register_models(self, turn_obs, depth_encode, policy):
        """Register model functions"""
        self.turn_obs = turn_obs
        self.depth_encode = depth_encode
        self.policy = policy


def load_models(logdir, device="cuda"):
    """Load the trained models"""
    rospy.loginfo(f"Loading models from: {logdir}")
    
    # Load configuration
    with open(osp.join(logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook=OrderedDict)
    
    # Update config for real robot
    config_dict["control"]["computer_clip_torque"] = True
    
    # Load base model
    base_model_name = 'base_jit.pt'
    base_model_path = os.path.join(logdir, base_model_name)
    base_model = torch.jit.load(base_model_path, map_location=device)
    base_model.eval()
    
    # Extract model components
    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone
    
    # Load vision model
    vision_model_name = 'vision_weight.pt'
    vision_model_path = os.path.join(logdir, vision_model_name)
    
    # Import here to avoid dependency if not using vision
    from rsl_rl.modules import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
    
    vision_model = torch.load(vision_model_path, map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()
    
    # Define observation processing function
    def turn_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        rospy.logdebug(f'yaw: {yaw}')
        
        proprio[:, 6:8] = yaw
        lin_vel_latent = estimator(proprio)
        
        activation = nn.ELU()
        priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))
        
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)
        return obs
    
    # Define depth encoding function
    def encode_depth(depth_image, proprio):
        depth_latent_yaw = depth_encoder(depth_image, proprio)
        if torch.isnan(depth_latent_yaw).any():
            rospy.logwarn('depth_latent_yaw contains nan')
        return depth_latent_yaw
    
    # Define policy function
    def actor_model(obs):
        action = actor(obs)
        return action
    
    return config_dict, turn_obs, encode_depth, actor_model


def main(args):
    # Load models
    config_dict, turn_obs, encode_depth, actor_model = load_models(args.logdir, device="cuda")
    
    # Create robot node
    env_node = Action(
        cfg=config_dict,
        model_device="cuda",
        dryrun=not args.nodryrun,
        mode=args.mode
    )
    
    # Register models
    env_node.register_models(turn_obs=turn_obs, depth_encode=encode_depth, policy=actor_model)
    
    # 模型预热
    env_node.warm_up()
    
    rospy.loginfo("Model and Policy are ready")
    
    if args.loop_mode == "timer":
        # Use ROS timer for main loop
        rospy.Timer(rospy.Duration(env_node.duration), env_node.main_loop)
        rospy.spin()
    elif args.loop_mode == "while":
        # 固定频率执行主循环
        rate = rospy.Rate(1.0 / env_node.duration)
        while not rospy.is_shutdown():
            start_time = time.monotonic()
            # print(f"main_loop duration: {env_node.duration}")
            env_node.main_loop()
            #固定在100Hz
            # 计算剩余时间
            remaining_time = env_node.duration - (time.monotonic() - start_time)
            if remaining_time > 0:
                rate.sleep()
            rospy.loginfo(f'loop duration: {time.monotonic() - start_time}')
    else:
        rospy.logerr(f"Unknown loop mode: {args.loop_mode}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default='/home/zzf/RL/unitree_rl/src/rl/traced', 
                       help="Directory containing config.json and model files")
    parser.add_argument("--nodryrun", action="store_true", default=True,
                       help="Disable dryrun mode")
    parser.add_argument("--loop_mode", type=str, default="timer",
                       choices=["while", "timer"],
                       help="Main loop execution mode")
    parser.add_argument("--mode", type=str, default="parkour",
                       choices=["parkour", "walk"])
    
    args = parser.parse_args()
    
    if args.logdir is None:
        parser.error("--logdir is required")
    
    try:
        main(args)
    except rospy.ROSInterruptException:
        pass