# import gym
# from gym import spaces
from pymunk import Space as pymunk_spaces

import genesis as gs
import ompl
import math
import torch
import json

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.ur5.pymunk_override import DrawOptions
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from taichi.examples.simulation.mass_spring_game import fixed
from gym import spaces
from Genesis.genesis import xyz_to_quat

genesis_cfg = {
    "urdf": "/home/natu/myproj/diffusion_policy/data/assets/ur5/ur5_robotiq85.urdf",
    "log_json": "/home/natu/myproj/diffusion_policy/outputs/ur5_log.json",
    "num_actions": 8,  # pos, quat, gripper
    # joint/link names
    "default_joint_angles": {  # [rad]
        'shoulder_pan_joint': -0.0,
        'shoulder_lift_joint': -0.9,
        'elbow_joint': -0.5,
        'wrist_1_joint': -1.4,
        'wrist_2_joint': -1.3,
        'wrist_3_joint': -0.3,
        'robotiq_85_left_knuckle_joint': 0.04,
        'robotiq_85_right_knuckle_joint': 0.04,
        'robotiq_85_left_inner_knuckle_joint': 0.04,
        'robotiq_85_right_inner_knuckle_joint': 0.04,
        'robotiq_85_left_finger_tip_joint': 0.04,
        'robotiq_85_right_finger_tip_joint': 0.04,
    },
    "joint_names": [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
        'robotiq_85_left_knuckle_joint',
        'robotiq_85_right_knuckle_joint',
        'robotiq_85_left_inner_knuckle_joint',
        'robotiq_85_right_inner_knuckle_joint',
        'robotiq_85_left_finger_tip_joint',
        'robotiq_85_right_finger_tip_joint',
    ],
    # PD
    "kp": [4500, 4500, 3500, 3500, 2000, 2000, 100, 100, 100, 100, 100, 100, ],
    "kd": [450, 450, 350, 350, 200, 200, 10, 10, 10, 10, 10, 10],
    "force_limit_l": [-87, -87, -87, -87, -87, -87, -12, -12, -12, -100, -100, -100],
    "force_limit_u": [87, 87, 87, 87, 87, 87, 12, 12, 12, 100, 100, 100],
    # termination

    # base pose
    "base_init_pos": [-0, -0.9, -0.5, -1.4, -1.3, -0.3, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
    "base_init_quat": [0.70710678, -0.70710678, 0.0, 0.0],

    "base_box_pos": [0.65, 0.0, 0.02],
    "base_box_quat": [1.0, 0, 0, 0],  # [x, y, z, w]
    "box_pos_randamin_range": 0.20,  # box position random range
    "episode_length_s": 20.0,
    "action_scale": 0.05,
    "action_scale_deg": 3.0,  # deg for each action
    "simulate_action_latency": True,
    "clip_actions": 1.0,
    "num_obs": 16,  # 3 for EE postion, 4 for EE quaternion, 3 for box posision, 6 for action
}


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom


class ur5Env():
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self):
        # for gym
        self.logging = True
        self.log_list = []
        self.log_file = genesis_cfg['log_json']
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0,
                          -1.0, -1.0, -1.0,
                          -1.0, -1.0, 0,
                          0], dtype=np.float64),
            high=np.array([1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0], dtype=np.float64),
            shape=(10,),
            dtype=np.float64
        )

        # positional goal for agent
        clip_actions = genesis_cfg["clip_actions"]
        self.action_space = spaces.Box(
            low=np.array([-clip_actions, -clip_actions, -clip_actions,
                          -clip_actions, -clip_actions, -clip_actions, -clip_actions], dtype=np.float64),
            high=np.array([clip_actions, clip_actions, clip_actions,
                           clip_actions, clip_actions, clip_actions, clip_actions], dtype=np.float64),
            shape=(7,),
            dtype=np.float64
        )

        # for genesis
        gs.init(logging_level="warning")
        num_envs = 1
        self.num_envs = num_envs
        self.num_obs = genesis_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = genesis_cfg["num_actions"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(genesis_cfg["episode_length_s"] / self.dt)

        self.env_cfg = genesis_cfg

        # self.obs_scales = obs_cfg["obs_scales"]
        self.reached_goal = torch.tensor(
            [False] * self.num_envs)  # flag to indicate if the goal is reached in any environment

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0, 0, 0.5),
                camera_fov=30,

            ),
            # vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=self.env_cfg["base_box_pos"],
            )
        )

        # add robot
        file = genesis_cfg["urdf"]
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=file,
                fixed=True,
            ),
        )
        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = list(np.arange(6))
        self.all_dof_idx = list(np.arange(12))

        # PD control parameters
        self.robot.set_dofs_kp(self.env_cfg["kp"], self.all_dof_idx)
        self.robot.set_dofs_kv(self.env_cfg["kd"], self.all_dof_idx)
        self.robot.set_dofs_force_range(
            self.env_cfg["force_limit_l"],
            self.env_cfg["force_limit_u"]
        )
        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        num_envs = self.num_envs
        qpos = torch.tensor(qpos, device=self.device, dtype=torch.float).repeat(num_envs, 1)
        self.robot.set_qpos(qpos, envs_idx=list(range(num_envs)))
        self.scene.step()
        self.target_qripper_closed = 0

        box_pos = self.cube.get_dofs_position()

        # prepare reward functions and multiply reward scales by dt

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        self.action = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=torch.float,
        )
        print("genesis initialized.")

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        num_envs = self.num_envs
        qpos = torch.tensor(qpos, device=self.device, dtype=gs.tc_float).repeat(num_envs, 1)
        self.robot.set_qpos(qpos, envs_idx=envs_idx)

        box_pos = torch.tensor(self.env_cfg["base_box_pos"], device=gs.device, dtype=gs.tc_float).repeat(num_envs, 1)
        box_quat = torch.tensor(self.env_cfg["base_box_quat"], device=gs.device, dtype=gs.tc_float).repeat(num_envs, 1)

        self.pox_pos = box_pos

        # box_posのxとyをランダムで±0.1の範囲に変化させる
        range = self.env_cfg["box_pos_randamin_range"]
        torch.manual_seed(5)
        rand_x = (range * 2 * torch.rand(len(envs_idx), device=gs.device) - range)
        box_pos[:, 0] += rand_x
        rand_y = (range * 2 * torch.rand(len(envs_idx), device=gs.device) - range)
        box_pos[:, 1] += rand_y

        print("box_pos", box_pos)


        self.cube.set_pos(box_pos, envs_idx=envs_idx)
        self.cube.set_quat(box_quat, envs_idx=envs_idx)

        # reset base
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        # self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.reached_goal = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        observation = self._get_obs()
        self.episode_length_buf[:] = 0
        return observation

    def step(self, actions):
        actions_t = torch.tensor(actions, device=self.device, dtype=gs.tc_float)
        self.actions = torch.clip(actions_t, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        for action in self.actions:
            qpos_all = self.robot.get_dofs_position(self.all_dof_idx)  # only use the first 6 dofs for control
            pos, quat = self.robot.forward_kinematics(qpos_all)
            eepos = pos[:, 6]  # shape: (num_envs, 3)
            eequat = quat[:, 6]  # shape: (num_envs, 4)

            self.target_qripper_closed = action[6]

            # eequatをdeg(rx, ry, rz)に変換
            ee_deg = quat_to_xyz(eequat)

            delta_pos = action * self.env_cfg["action_scale"]
            target_eepos = eepos + delta_pos[0:3]
            target_eedeq = ee_deg + delta_pos[3:6] * self.env_cfg["action_scale_deg"]

            target_eequat = xyz_to_quat(target_eedeq)

            target_dof_pos = self.robot.inverse_kinematics(
                link=self.robot.get_link("wrist_3_link"),
                pos=target_eepos,
                quat=target_eequat,
                respect_joint_limit=True,
                dofs_idx_local=self.motors_dof_idx,
            )

            self.robot.set_qpos(target_dof_pos[:, 0:6], self.motors_dof_idx)
            self.robot.zero_all_dofs_velocity()

            box_pos = self.cube.get_pos()
            self.episode_length_buf += 1

            if self.logging:
                action = action.detach().cpu().numpy().tolist()
                qpos = qpos_all.detach().cpu().numpy().tolist()
                box_pos = box_pos[0].detach().cpu().numpy().tolist()
                is_gripper_closed = self.target_qripper_closed.detach().cpu().numpy().tolist()
                eepos_list = eepos[0].detach().cpu().numpy().tolist()
                ee_to_box = [eepos_list[0] - box_pos[0], eepos_list[1] - box_pos[1], eepos_list[2] - box_pos[2]]
                self.log_list.append({
                    "eepos" : eepos[0].detach().cpu().numpy().tolist(),
                    "eequat" : eequat[0].detach().cpu().numpy().tolist(),
                    "action": action,
                    "qpos": qpos,
                    "box_pos": box_pos,
                    "ee_to_box": ee_to_box,
                    "is_gripper_closed" : is_gripper_closed,
                    "episode_length": int(self.episode_length_buf[0].detach().cpu())
                })


            self.scene.step()


        # NumPyに変換
        obs_buf_numpy = self._get_obs()

        reward = 0.0
        done = 0
        episode_len = int(self.episode_length_buf[0].detach().cpu())
        if self.target_qripper_closed > 0.5:
            done = 1
            reward = 1.0
        elif episode_len > self.max_episode_length:
            done = 1
            reward = -1.0

        if self.logging and done == 1:
            with open(self.log_file, "w") as f:
                j = {
                    "data" : self.log_list,
                }
                f.write(json.dumps(j, indent=4))

        return obs_buf_numpy, reward, done, {}

    def _get_obs(self):
        obs = self._obs_numpy()
        return obs

    def _obs_numpy(self):
        box_pos = self.cube.get_pos()

        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        pos, quat = self.robot.forward_kinematics(qpos)
        eepos = pos[:, 6]
        eequat = quat[:, 6]
        target_qripper_closed = torch.tensor([[self.target_qripper_closed]], device=self.device, dtype=gs.tc_float)

        ee_xyz = quat_to_xyz(eequat)

        ee_to_box = eepos - box_pos

        # compute observations
        self.obs_buf = torch.cat([
                eepos,  # 3
                ee_xyz,  # 3
                box_pos,  # 3
                #ee_to_box,
                target_qripper_closed,
                torch.ones((self.num_envs, 10), device=self.device, dtype=torch.float) #mask
            ],
            axis=-1,
        )

        obs = self.obs_buf.cpu().numpy()

        return obs