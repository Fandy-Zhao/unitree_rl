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
        self.duration = 0.02
        
        print("Action init done")
        
        self.start_ros_handlers()
        
    # def __init__(self, cfg, *args, **kwargs, model_device="cuda", dryrun=False, mode="parkour"):
    #     super().__init__(*args, robot_class_name= "Go2", **kwargs)
        
    #     rospy.init_node('rl_action', anonymous=True)
        
    #     self.model_device = model_device
    #     self.dryrun = dryrun
    #     self.mode = mode
        
    #     # Load configuration
    #     self.cfg = cfg
    #     self.dt = self.cfg["sim"]["dt"] if "sim" in self.cfg else 0.02
    #     self.decimation = self.cfg["control"]["decimation"] if "control" in self.cfg else 1
    #     self.duration = self.dt * self.decimation
        
    #     # Control gains
    #     self.p_gains = np.array(cfg["control"]["stiffness"], dtype=np.float32) if "control" in cfg and "stiffness" in cfg["control"] else np.ones(12, dtype=np.float32) * 20.0
    #     self.d_gains = np.array(cfg["control"]["damping"], dtype=np.float32) if "control" in cfg and "damping" in cfg["control"] else np.ones(12, dtype=np.float32) * 0.5
        
    #     # Initialize state variables
    #     self.low_state = LowState()
    #     self.low_cmd = LowCmd()
    #     self.joy_stick_buffer = None
    #     self.depth_image = None
        
    #     # Timing and counters
    #     self.global_counter = 0
    #     self.visual_update_interval = 5
    #     self.sim_ite = 3
        
    #     # Policy flags
    #     self.use_stand_policy = False
    #     self.use_parkour_policy = False
    #     self.use_sport_mode = True
        
    #     # Load simulation actions
    #     self.actions_sim = torch.from_numpy(np.load('Action_sim_335-11_flat.npy')).to(self.model_device)
        
    #     # Initialize ROS publishers and subscribers
    #     self.setup_ros_handlers()
        
    #     # Initialize observation buffers
    #     self.n_proprio = 48  # Adjust based on your actual observation dimensions
    #     self.n_depth_latent = 512  # Adjust based on your model
    #     self.n_hist_len = 10  # Adjust based on your model
    #     self.reset_obs()
        
    #     rospy.loginfo("RL Node initialized")
        
    # def setup_ros_handlers(self):
    #     """Setup ROS publishers and subscribers"""
    #     # Publisher for low-level commands
    #     self.cmd_pub = rospy.Publisher('/low_cmd', LowCmd, queue_size=1)
        
    #     # Subscriber for low-level state
    #     #由state_rl单独发送
    #     rospy.Subscriber('/rl/low_state', LowState, self.low_state_callback)
        
    #     # Subscriber for depth image
    #     rospy.Subscriber('/depth/image', Image, self.depth_image_callback)
        
    #     # Subscriber for joystick (you'll need to define this message type)
    #     # rospy.Subscriber('/joy', Joy, self.joy_callback)
        
    #     # Initialize CV bridge for image conversion
    #     self.bridge = CvBridge()
        
    # def low_state_callback(self, msg):
    #     """Callback for low state messages"""
    #     # self.get_logger().warn("Low state message received.")
    #     """ store and handle proprioception data """
    #     self.low_state_buffer = msg # keep the latest low state

    #     ################### refresh dof_pos and dof_vel ######################
    #     # for sim_idx in range(self.NUM_DOF):
    #     #     real_idx = self.dof_map[sim_idx]
    #     #     self.dof_pos_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.dof_signs[sim_idx]
    #     # for sim_idx in range(self.NUM_DOF):
    #     #     real_idx = self.dof_map[sim_idx]
    #     #     self.dof_vel_[0, sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.dof_signs[sim_idx]
        
    # def depth_image_callback(self, msg):
    #     """Callback for depth image messages"""
    #     try:
    #         # Convert ROS image to OpenCV image
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
    #         # Normalize or process the depth image as needed
    #         self.depth_image = torch.from_numpy(cv_image).float().to(self.model_device)
    #         self.depth_image = self.depth_image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    #     except Exception as e:
    #         rospy.logerr(f"Error processing depth image: {e}")
            
    # def joy_callback(self, msg):
    #     """Callback for joystick messages"""
    #     # Process joystick message based on your message format
    #     # This is a placeholder - you'll need to implement based on your actual joystick message
    #     self.joy_stick_buffer = msg
        
    # def get_proprio(self):
    #     start_time = time.monotonic()

    #     ang_vel = self._get_ang_vel_obs()  # (1, 3)
    #     ang_vel_time = time.monotonic()

    #     imu = self._get_imu_obs()  # (1, 2)
    #     imu_time = time.monotonic()

    #     yaw_info = self._get_delta_yaw_obs()  # (1, 3)
    #     yaw_time = time.monotonic()

    #     commands = self._get_commands_obs()  # (1, 3)
    #     commands_time = time.monotonic()

    #     if self.mode == "parkour":
    #         parkour_walk = torch.tensor([[1, 0]], device= self.model_device, dtype= torch.float32) # parkour
    #     elif self.mode == "walk":
    #         parkour_walk = torch.tensor([[0, 1]], device= self.model_device, dtype= torch.float32) # walk

    #     dof_pos = self._get_dof_pos_obs()  # (1, 12)
    #     dof_pos_time = time.monotonic()

    #     dof_vel = self._get_dof_vel_obs()  # (1, 12)
    #     dof_vel_time = time.monotonic()

    #     last_actions = self._get_last_actions_obs().view(1, -1)  # (1, 12)
    #     last_action_time = time.monotonic()

    #     contact = self._get_contact_filt_obs()  # (1, 4)
    #     contact_time = time.monotonic()
        
    #     proprio = torch.cat([ang_vel, imu, yaw_info, commands, parkour_walk,
    #                     dof_pos, dof_vel,
    #                     last_actions, 
    #                     contact], dim=-1)

    #     self.proprio_history_buf = torch.where(
    #         (self.episode_length_buf <= 1)[:, None, None], 
    #         torch.stack([proprio] * self.n_hist_len, dim=1),
    #         torch.cat([
    #             self.proprio_history_buf[:, 1:],
    #             proprio.unsqueeze(1)
    #         ], dim=1)
    #     )
    #     end_time = time.monotonic()

    #     # print('ang vel time: {:.5f}'.format(ang_vel_time - start_time),
    #     #         'imu time: {:.5f}'.format(imu_time - ang_vel_time),
    #     #         'yaw time: {:.5f}'.format(yaw_time - imu_time),
    #     #         'command time: {:.5f}'.format(commands_time - yaw_time),
    #     #         'dof pos time: {:.5f}'.format(dof_pos_time - commands_time),
    #     #         'dof vel time: {:.5f}'.format(dof_vel_time - dof_pos_time),
    #     #         'last action time: {:.5f}'.format(last_action_time - dof_vel_time),
    #     #         'contact time: {:.5f}'.format(contact_time - last_action_time)
    #     #         )
        
    #     self.episode_length_buf += 1

    #     return proprio

    # def _get_history_proprio(self):
    #     """Get history of proprioceptive observations"""
    #     # This is a placeholder - implement based on your history buffer
    #     return torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.model_device)
    
    # def _get_depth_image(self):
    #     """Get current depth image"""
    #     if self.depth_image is None:
    #         # Return a zero image if no depth image received yet
    #         return torch.zeros(1, 1, 58, 87, device=self.model_device)  # Adjust dimensions based on your model
    #     return self.depth_image
    
    # def send_action(self, action):
    #     """Send action commands to robot"""
    #     # Convert action tensor to numpy array
    #     if isinstance(action, torch.Tensor):
    #         action = action.cpu().numpy().flatten()
        
    #     # Create low command
    #     low_cmd = LowCmd()
    #     low_cmd.header = Header()
    #     low_cmd.header.stamp = rospy.Time.now()
        
    #     # Fill motor commands (12 motors)
    #     for i in range(12):
    #         motor_cmd = MotorCmd()
    #         motor_cmd.mode = 0x0A  # Position mode
    #         motor_cmd.q = action[i] if i < len(action) else 0.0
    #         motor_cmd.dq = 0.0
    #         motor_cmd.tau = 0.0
    #         motor_cmd.Kp = self.p_gains[i]
    #         motor_cmd.Kd = self.d_gains[i]
    #         low_cmd.motorCmd[i] = motor_cmd
            
    #     # Publish command
    #     if not self.dryrun:
    #         self.cmd_pub.publish(low_cmd)
    #     else:
    #         rospy.loginfo(f"Dry run - Action: {action[:4]}...")  # Log first 4 actions
            
    # def send_stand_action(self, action):
    #     """Send stand action (placeholder)"""
    #     self.send_action(action)
        
    # def get_stand_action(self):
    #     """Get stand action (placeholder)"""
    #     return torch.zeros(12, device=self.model_device)
    
    # def _sport_mode_change(self, mode_id):
    #     """Change sport mode (placeholder)"""
    #     rospy.loginfo(f"Changing sport mode to: {mode_id}")
        
    # def _sport_state_change(self, state):
    #     """Change sport state (placeholder)"""
    #     rospy.loginfo(f"Changing sport state to: {state}")
        
    # def reset_obs(self):
    #     """Reset observation buffers"""
    #     self.proprio_history = []
    #     self.last_depth_image = None
    #     self.depth_latent_yaw = None
       
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
            rospy.logdebug(f'Action: {action}')
            
            publish_time = time.time()
            
            rospy.loginfo(f"Loop timings: "
                         f"get proprio: {get_pro_time - start_time:.5f}, "
                         f"get hist pro: {get_hist_pro_time - get_pro_time:.5f}, "
                         f"get_depth: {get_obs_time - get_hist_pro_time:.5f}, "
                         f"get obs: {get_obs_time - start_time:.5f}, "
                         f"turn_obs: {turn_obs_time - get_obs_time:.5f}, "
                         f"policy: {policy_time - turn_obs_time:.5f}, "
                         f"total: {publish_time - start_time:.5f}")
                         
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
    
    # Warm up
    env_node.warm_up()
    
    rospy.loginfo("Model and Policy are ready")
    
    if args.loop_mode == "timer":
        # Use ROS timer for main loop
        rospy.Timer(rospy.Duration(env_node.duration), env_node.main_loop)
        rospy.spin()
    elif args.loop_mode == "while":
        # Use while loop with rate control
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