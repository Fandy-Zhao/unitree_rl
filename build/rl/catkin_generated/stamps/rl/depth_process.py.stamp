#!/usr/bin/env python3
# 注意: 在ROS1中，此文件通常需放在ROS包的`scripts`目录下，并赋予可执行权限(chmod +x)

import rospy  # 修改1: 导入ROS1核心库
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo
# 修改2: 导入ROS1版本的Unitree包，包名可能需要根据实际情况调整
# from unitree_ros import UnitreeRosReal

import os, sys
import os.path as osp
import json
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# from rsl_rl import modules

import pyrealsense2 as rs
import cv2
import ros_numpy   
import ros_numpy as rnp
import random
from collections import deque

# 以下部分与硬件和自定义模块相关，通常无需修改
if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "x86",
    ))
elif os.uname().machine == "aarch64":
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "aarch64",
    ))


@torch.no_grad()
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data


class VisualHandlerNode:
    """ A wrapper class for the realsense camera """
    def __init__(self,
            cfg: dict,
            cropping: list = [0, 0, 0, 0], # top, bottom, left, right
            rs_resolution: tuple = (480, 270), # width, height for the realsense camera)
            rs_fps: int= 30,
            depth_image_topic="/camera_face/depth/image_raw",  # 仿真器发布的深度图话题名
            depth_input_topic= "/camera/forward_depth",
            camera_info_topic= "/camera/camera_info",
            forward_depth_image_topic= "/forward_depth_image",
        ):
        
        rospy.init_node("depth_image", anonymous=True)
        
        self.cfg = cfg
        self.cropping = cropping
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.depth_image_topic = depth_image_topic
        self.depth_input_topic = depth_input_topic
        self.camera_info_topic = camera_info_topic
        self.forward_depth_image_topic = forward_depth_image_topic

        self.depth_image = None
        self.original_size = None
        
        self.parse_args()
        # self.start_pipeline()
        self.start_ros_handlers()

    def parse_args(self):
        self.output_resolution = [58, 87]
        depth_range = [0.0, 3.0]
        self.depth_range = (depth_range[0], depth_range[1] * 1000) # [m] -> [mm]

    def start_pipeline(self):
        
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.depth,
            self.rs_resolution[0],
            self.rs_resolution[1],
            rs.format.z16,
            self.rs_fps,
        )
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        self.rs_align = rs.align(rs.stream.depth)

        self.rs_hole_filling_filter = rs.hole_filling_filter()
        self.rs_spatial_filter = rs.spatial_filter()
        self.rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_spatial_filter.set_option(rs.option.holes_fill, 4)
        self.rs_temporal_filter = rs.temporal_filter()
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_filters = [
            self.rs_hole_filling_filter,
            self.rs_spatial_filter,
            self.rs_temporal_filter,
        ]

    def start_ros_handlers(self):
        
        self.depth_image_sub = rospy.Subscriber(
            self.depth_image_topic,
            Image,
            self.depth_image_callback,
            queue_size=1
        )
        
        self.depth_input_pub = rospy.Publisher(
            self.depth_input_topic,
            Image,
            queue_size=1  
        )

        self.forward_depth_image_pub = rospy.Publisher(
            self.forward_depth_image_topic,
            Float32MultiArray,
            queue_size=1
        )
        
        
        rospy.loginfo("ros handlers started")

    def depth_image_callback(self, msg):
        if msg.encoding != '32FC1':
            rospy.logwarn_throttle(5.0, f"期望编码32FC1，但收到 {msg.encoding}。转换可能出错。")
        
        self.depth_image = ros_numpy.numpify(msg)
        
        # 打印信息用于调试（首次或节流）
        if self.depth_image is not None:
            rospy.loginfo_once(f"成功获取深度图。尺寸: {self.depth_image.shape}， 类型: {self.depth_image.dtype}， 范围: [{self.depth_image.min():.2f}, {self.depth_image.max():.2f}] m")

    @staticmethod
    def calculate_center_crop(image_height, image_width, target_height=58, target_width=87):
        """
        计算居中裁剪区域，保持目标比例
        
        Args:
            image_height: 输入图像高度
            image_width: 输入图像宽度
            target_height: 目标高度 (58)
            target_width: 目标宽度 (87)
        
        Returns:
            (top, bottom, left, right) 裁剪参数
        """
        # 计算目标比例
        target_ratio = target_height / target_width  # 58/87 ≈ 0.6667
        
        # 当前图像比例
        current_ratio = image_height / image_width
        
        if current_ratio > target_ratio:
            # 图像"太高"，需要裁剪上下
            # 保持宽度不变，按比例计算高度
            crop_height = int(image_width * target_ratio)
            top = (image_height - crop_height) // 2
            bottom = image_height - crop_height - top
            left = 0
            right = 0
        else:
            # 图像"太宽"，需要裁剪左右
            # 保持高度不变，按比例计算宽度
            crop_width = int(image_height / target_ratio)
            left = (image_width - crop_width) // 2
            right = image_width - crop_width - left
            top = 0
            bottom = 0
        
        return top, bottom, left, right

    def apply_cropping(self, depth_tensor, crop_params):
            """
            应用裁剪到深度张量
            
            Args:
                depth_tensor: PyTorch张量，形状 (1, 1, H, W)
                crop_params: (top, bottom, left, right)
            
            Returns:
                裁剪后的张量
            """
            top, bottom, left, right = crop_params
            h, w = depth_tensor.shape[2], depth_tensor.shape[3]
            
            # 验证裁剪参数有效性
            if top + bottom < h and left + right < w:
                return depth_tensor[:, :, top:h-bottom, left:w-right]
            else:
                rospy.logwarn(f"裁剪参数无效: {crop_params}，图像尺寸: ({h}, {w})")
                return depth_tensor
            
    def get_depth_frame_sim(self, use_dynamic_crop=True):
        
        if self.depth_image is None:
            rospy.logwarn("深度图像数据为空")
            return None
    
        try:
            # 1. 转换为NumPy数组
            depth_image_np = self.depth_image
            
            # 记录原始尺寸（仅第一次）
            if self.original_size is None:
                self.original_size = (depth_image_np.shape[0], depth_image_np.shape[1])
                rospy.loginfo(f"原始图像尺寸: {self.original_size}")
                rospy.loginfo(f"图像数据类型: {depth_image_np.dtype}, 范围: [{depth_image_np.min():.3f}, {depth_image_np.max():.3f}]")
            
            # 2. 处理NaN/无穷大值
            depth_image_np = np.nan_to_num(
                depth_image_np, 
                nan=0.0, 
                posinf=self.depth_range[1], 
                neginf=self.depth_range[0]
            )
            
            # 3. 转换为PyTorch张量
            # 注意：深度图应该是单通道，所以unsqueeze(0).unsqueeze(0) 添加批次和通道维度
            depth_tensor = torch.from_numpy(depth_image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            
            # 记录初始张量形状
            rospy.logdebug_once(f"初始张量形状: {depth_tensor.shape}")
            
            # 4. 计算并应用裁剪
            if use_dynamic_crop:
                # 动态计算裁剪参数
                crop_params = self.calculate_center_crop(
                    depth_tensor.shape[2], 
                    depth_tensor.shape[3]
                )
                rospy.logdebug_once(f"动态裁剪参数: {crop_params}")
            else:
                # 使用静态裁剪参数
                crop_params = self.static_crop
            
            # 应用裁剪
            if any(crop_params):  # 如果有任何非零裁剪参数
                depth_tensor = self.apply_cropping(depth_tensor, crop_params)
                rospy.loginfo_once(f"应用裁剪: {crop_params}，裁剪后尺寸: {depth_tensor.shape[2:]}")
            
            # 5. 归一化到深度范围 [0, 1]
            depth_tensor = torch.clamp(depth_tensor, self.depth_range[0], self.depth_range[1])
            depth_tensor = (depth_tensor - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
            
            # 6. 缩放到目标尺寸 (58, 87)
            # 使用双线性插值保持空间关系
            depth_tensor = F.interpolate(
                depth_tensor, 
                size=self.output_resolution, 
                mode='bilinear', 
                align_corners=False
            )
            rospy.logdebug_once(f"缩放后张量形状: {depth_tensor.shape}")
            
            # 7. 【修复1】先保存用于发布的数据（归一化后，但未减去0.5的数据）
            # 用于发布的深度数据（反归一化到原始范围）
            depth_for_publish = depth_tensor.clone()
            depth_for_publish = depth_for_publish * (self.depth_range[1] - self.depth_range[0]) + self.depth_range[0]
            
            # 【修复2】确保是2D数组 (58, 87)
            depth_for_publish_np = depth_for_publish.squeeze().cpu().numpy().astype(np.float32)
            
            # 验证数组形状
            if depth_for_publish_np.ndim != 2:
                rospy.logwarn(f"发布数据维度错误: {depth_for_publish_np.shape}，期望 (58, 87)")
                # 尝试修复：如果是3D且第三维是1，则压缩
                if depth_for_publish_np.ndim == 3 and depth_for_publish_np.shape[2] == 1:
                    depth_for_publish_np = depth_for_publish_np[:, :, 0]
                else:
                    # 尝试重塑
                    depth_for_publish_np = depth_for_publish_np.reshape(self.output_resolution[0], self.output_resolution[1])
            
            # 8. 发布处理后的深度图（用于可视化/调试）
            try:
                depth_input_msg = rnp.msgify(Image, depth_for_publish_np, encoding="32FC1")
                depth_input_msg.header.stamp = rospy.Time.now()
                depth_input_msg.header.frame_id = "d435_sim_depth_link"
                self.depth_input_pub.publish(depth_input_msg)
                rospy.logdebug_once("深度图输入已发布")
            except Exception as e:
                rospy.logerr(f"发布深度图时出错: {e}")
                rospy.logerr(f"数组形状: {depth_for_publish_np.shape}, 维度: {depth_for_publish_np.ndim}")
            
            # 9. 调整数值范围到 [-0.5, 0.5]（模型需要的输入范围）
            depth_tensor = depth_tensor - 0.5
            
            # 10. 记录处理统计信息
            rospy.loginfo_throttle(5.0, 
                f"深度图处理: 原始{self.original_size} → 裁剪后{depth_tensor.shape[2:]} → 输出{self.output_resolution}"
            )
            
            return depth_tensor
            
        except Exception as e:
            rospy.logerr(f"深度图处理错误: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None
    
    # def get_depth_frame(self):
    #     # read from pyrealsense2, preprocess and write the model embedding to the buffer
    #     latency_range  = [0.08, 0.142]
    #     rs_frame = self.rs_pipeline.wait_for_frames(int( latency_range[1] * 1000 )) # ms
        
    #     depth_frame = rs_frame.get_depth_frame()
    #     if not depth_frame:
    #         # 修改7: ROS1的节流日志函数
    #         rospy.logerr_throttle(60, "No depth frame") # throttle_duration_sec改为秒
    #         return None
        
    #     for rs_filter in self.rs_filters:
    #         depth_frame = rs_filter.process(depth_frame)
        
    #     depth_image_pyt = torch.from_numpy(np.asanyarray(depth_frame.get_data()).astype(np.float32)).unsqueeze(0)
        
    #     # apply torch filters
    #     # 注意：切片操作需确保索引有效，当crop值为0时，-0-1的写法会导致问题，已调整
    #     crop_top = self.cropping[0]
    #     crop_bottom = self.cropping[1]
    #     crop_left = self.cropping[2]
    #     crop_right = self.cropping[3]
        
    #     h_start = crop_top
    #     h_end = -crop_bottom if crop_bottom > 0 else None
    #     w_start = crop_left
    #     w_end = -crop_right if crop_right > 0 else None
        
    #     depth_image_pyt = depth_image_pyt[:, h_start:h_end, w_start:w_end]

    #     depth_image_pyt = torch.clip(depth_image_pyt, self.depth_range[0], self.depth_range[1]) / (self.depth_range[1] - self.depth_range[0])
    #     depth_image_pyt = resize2d(depth_image_pyt, self.output_resolution)

    #     # publish the depth image input to ros topic
    #     rospy.loginfo_once("depth range: {}-{}".format(self.depth_range[0], self.depth_range[1]))
    #     depth_input_data = (
    #         depth_image_pyt.detach().cpu().numpy() * (self.depth_range[1] - self.depth_range[0]) + self.depth_range[0]).astype(np.uint16)[0]
        
    #     depth_image_pyt -= 0.5 # [-0.5, 0.5])

    #     # 使用ros_numpy转换，注意编码
    #     depth_input_msg = rnp.msgify(Image, depth_input_data.astype(np.float32), encoding="32FC1")
    #     # 修改8: 获取ROS1时间戳
    #     depth_input_msg.header.stamp = rospy.Time.now()
    #     depth_input_msg.header.frame_id = "d435_sim_depth_link"
    #     self.depth_input_pub.publish(depth_input_msg)
    #     rospy.loginfo_once("depth input published")

    #     return depth_image_pyt

    def publish_depth_data(self, depth_data):
        msg = Float32MultiArray()
        msg.data = depth_data.flatten().detach().cpu().numpy().tolist()
        self.forward_depth_image_pub.publish(msg)
        rospy.loginfo_once("depth data published")

    def start_main_loop_timer(self, duration):
        
        self.main_loop_timer = rospy.Timer(rospy.Duration(duration), self.main_loop_callback)

    def main_loop_callback(self, event):
        
        depth_image_pyt = self.get_depth_frame_sim()
        if depth_image_pyt is not None:
            self.publish_depth_data(depth_image_pyt)
        else:
            rospy.logwarn("One frame of depth latent if not acquired")

    def main_loop(self):
        """用于'while'循环模式的主循环函数"""
        depth_image_pyt = self.get_depth_frame_sim()
        if depth_image_pyt is not None:
            self.publish_depth_data(depth_image_pyt)
            #转换成图像
            depth_image = depth_image_pyt[0].numpy() * 255
            cv2.imshow("depth image", depth_image.astype(np.uint8))
            cv2.waitKey(10)
        else:
            rospy.logwarn("One frame of depth latent if not acquired")


@torch.inference_mode()
def main(args):

    assert args.logdir is not None, "Please provide a logdir"
    config_path = osp.join(args.logdir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f, object_pairs_hook=OrderedDict)
    print(config_dict)
        
    device = "cpu"
    duration = 0.01

    visual_node = VisualHandlerNode(
        cfg=json.load(open(config_path, "r")),
        cropping=[args.crop_top, args.crop_bottom, args.crop_left, args.crop_right],
        rs_resolution=(args.width, args.height),
        rs_fps=args.fps,
    )

    if args.loop_mode == "while":
        rospy.loginfo("Starting main loop (while mode)")
        rate = rospy.Rate(1.0 / duration)  # 使用rospy.Rate控制频率
        while not rospy.is_shutdown():     
            start_time = time.monotonic()
            visual_node.main_loop()
            rate.sleep()
    elif args.loop_mode == "timer":
        rospy.loginfo("Starting main loop (timer mode)")
        visual_node.start_main_loop_timer(duration)
        
        rospy.spin()

    
    rospy.loginfo("Shutting down depth camera node...")
    # visual_node.rs_pipeline.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", type=str, default=None, help="The directory which contains the config.json and model_*.pt files")
    
    parser.add_argument("--height",
        type=int,
        default=480,
        help="The height of the realsense image",
    )
    parser.add_argument("--width",
        type=int,
        default=640,
        help="The width of the realsense image",
    )
    parser.add_argument("--fps",
        type=int,
        default=30,
        help="The fps request to the rs pipeline",
    )
    parser.add_argument("--crop_left",
        type=int,
        default=80,
        help="num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_right",
        type=int,
        default=36,
        help="num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_top",
        type=int,
        default=60,
        help="num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--crop_bottom",
        type=int,
        default=100,
        help="num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument("--loop_mode", type=str, default="timer",
        choices=["while", "timer"],
        help="Select which mode to run the main policy control iteration",
    )

    args = parser.parse_args()
    main(args)
