# import gym
# from gym import spaces
from pymunk import Space as pymunk_spaces

import genesis as gs
import ompl
import math
import torch

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
    "urdf" : "/home/natu/myproj/diffusion_policy/data/assets/ur5/ur5_robotiq85.urdf",
    "log_csv" : "/home/natu/myproj/diffusion_policy/outputs/ur5_log.csv",
    "num_actions": 6,  # Δx、Δθ
    # joint/link names
    "default_joint_angles": {  # [rad]
        'shoulder_pan_joint' : -0.0,
        'shoulder_lift_joint' : -0.9,
        'elbow_joint'  : -0.5,
        'wrist_1_joint' :  -1.4,
        'wrist_2_joint' : -1.3,
        'wrist_3_joint' : -0.3,
        'robotiq_85_left_knuckle_joint' : 0.04,
        'robotiq_85_right_knuckle_joint' : 0.04,
        'robotiq_85_left_inner_knuckle_joint' : 0.04,
        'robotiq_85_right_inner_knuckle_joint' : 0.04,
        'robotiq_85_left_finger_tip_joint' : 0.04,
        'robotiq_85_right_finger_tip_joint' : 0.04,
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
    "kp": [4500, 4500, 3500, 3500, 2000, 2000, 100, 100, 100, 100, 100, 100,],
    "kd": [450,   450,  350,  350,  200,  200, 10, 10, 10, 10, 10, 10],
    "force_limit_l": [-87, -87, -87, -87, -87, -87, -12, -12, -12, -100, -100, -100],
    "force_limit_u": [ 87,  87,  87,  87,  87,  87,  12,  12,  12,  100,  100,  100],
    # termination

    # base pose
    "base_init_pos": [  -0, -0.9,  -0.5,  -1.4,  -1.3,  -0.3, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
    "base_init_quat": [0.70710678, -0.70710678, 0.0, 0.0],

    "base_box_pos" : [0.65, 0.0, 0.02],
    "base_box_quat" : [1.0, 0, 0, 0],  # [x, y, z, w]
    "box_pos_randamin_range": 0.20,  # box position random range
    "episode_length_s": 2.0,
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
        self.observation_space = spaces.Box(
            low=np.array([-1.0,-1.0,0,
                                -1.0,-1.0, -1.0,-1.0,
                                -1.0, -1.0, 0,
                                0], dtype=np.float64),
            high=np.array([1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0], dtype=np.float64),
            shape=(11,),
            dtype=np.float64
        )

        # positional goal for agent
        clip_actions = genesis_cfg["clip_actions"]
        self.action_space = spaces.Box(
            low=np.array([-clip_actions,-clip_actions,-clip_actions,-clip_actions,
                          -clip_actions,-clip_actions,-clip_actions,-clip_actions], dtype=np.float64),
            high=np.array([clip_actions, clip_actions, clip_actions, clip_actions,
                           clip_actions,clip_actions,clip_actions,clip_actions], dtype=np.float64),
            shape=(8,),
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
        self.n_steps = 8

        # self.obs_scales = obs_cfg["obs_scales"]
        self.reached_goal = torch.tensor([False] * self.num_envs)  # flag to indicate if the goal is reached in any environment

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0, 0, 0.5),
                camera_fov=30,

            ),
            #vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
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
                fixed=True,
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

        box_pos = self.cube.get_dofs_position()
        print("box_pos", box_pos)

        # prepare reward functions and multiply reward scales by dt

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=torch.float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        self.flog = open(genesis_cfg["log_csv"], "w")
        print("genesis initialized.")



    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos[0:6]
        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        num_envs = self.num_envs
        qpos = torch.tensor(qpos, device=self.device, dtype=gs.tc_float).repeat(len(envs_idx), 1)
        self.robot.set_qpos(qpos, envs_idx=envs_idx)

        box_pos = torch.tensor(self.env_cfg["base_box_pos"], device=gs.device, dtype=gs.tc_float).repeat(len(envs_idx), 1)
        box_quat = torch.tensor(self.env_cfg["base_box_quat"], device=gs.device, dtype=gs.tc_float).repeat(len(envs_idx), 1)

        # box_posのxとyをランダムで±0.1の範囲に変化させる
        range = self.env_cfg["box_pos_randamin_range"]
        rand_x = (range * 2 * torch.rand(len(envs_idx), device=gs.device) - range)
        box_pos[:, 0] += rand_x
        rand_y = (range * 2 * torch.rand(len(envs_idx), device=gs.device) - range)
        box_pos[:, 1] += rand_y

        self.cube.set_pos(box_pos, envs_idx=envs_idx)
        self.cube.set_quat(box_quat, envs_idx=envs_idx)

        # reset base
        self.robot.zero_all_dofs_velocity(envs_idx)


        # reset buffers
        self.last_actions[envs_idx] = 0.0
        # self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = False
        self.reached_goal[envs_idx] = False



    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        observation = self._get_obs()
        self.episode_length_buf[:] = 0
        return observation

    def step(self, actions):
        actions_t = torch.tensor(actions, device=self.device, dtype=gs.tc_float)
        self.actions = torch.clip(actions_t, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        if actions is not None:
            for i in range(self.n_steps):

                qpos_all = self.robot.get_dofs_position(self.all_dof_idx) # only use the first 6 dofs for control
                pos, quat = self.robot.forward_kinematics(qpos_all)
                eepos = pos[:, 6]  # shape: (num_envs, 3)
                eequat = quat[:, 6]  # shape: (num_envs, 4)

                # eequatをdeg(rx, ry, rz)に変換
                ee_deg = quat_to_xyz(eequat)

                delta_pos = self.actions[i] * self.env_cfg["action_scale"]
                target_eepos = eepos + delta_pos[0:3]
                target_eedeq = ee_deg + delta_pos[3:6] *  self.env_cfg["action_scale_deg"]

                target_eequat = xyz_to_quat(target_eedeq)

                target_dof_pos = self.robot.inverse_kinematics(
                    link=self.robot.get_link("wrist_3_link"),
                    pos=target_eepos,
                    quat=target_eequat,
                    respect_joint_limit=True,
                    dofs_idx_local=self.motors_dof_idx,
                )

                self.robot.set_qpos(target_dof_pos[:,0:6], self.motors_dof_idx)
                self.robot.zero_all_dofs_velocity()

                box_pos = self.cube.get_pos()
                self.episode_length_buf += 1

                logs = str(self.actions[i].detach().cpu().numpy().tolist()
                            + qpos_all.detach().cpu().numpy().tolist()
                            + box_pos.detach().cpu().numpy().tolist()) + "\n"
#                print(logs)
                self.flog.write(logs)

                self.scene.step()

        # update buffers
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.reached_goal

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())


        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        pos, quat = self.robot.forward_kinematics(qpos)
        eepos = pos[:,6]
        eequat = quat[:,6]

        # compute observations
        self.obs_buf = torch.cat(
            [
                eepos,  # 3
                eequat,  # 4
                box_pos, # 3
                torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float),
                torch.ones((self.num_envs, 11), device=self.device, dtype=torch.float),
            ],
            axis=-1,
        )


        self.last_actions = self.actions

        # obs, reward, done, info
        gripper = actions[0][7]
        #observation = np.stack((self.obs_buf_prev.cpu().numpy(), self.obs_buf.cpu().numpy()), axis=0)
        self.obs_buf = torch.cat(
            [
                eepos,  # 3
                eequat,  # 4
                box_pos,  # 3
                torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float),
                torch.ones((self.num_envs, 11), device=self.device, dtype=torch.float),
            ],
            axis=-1,
        )

        # NumPyに変換
        obs_buf_numpy = self.obs_buf.cpu().numpy()

        ep = int(self.episode_length_buf[0].detach().cpu())
        if ep > 90:
            done = 1
        else:
            done = 0

        return obs_buf_numpy, 1.0, done, {}


    def render(self, mode):
        pass
        # return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        box_pos = self.cube.get_pos()

        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        pos, quat = self.robot.forward_kinematics(qpos)
        eepos = pos[:,6]
        eequat = quat[:,6]


        # compute observations
        self.obs_buf = torch.cat([
                eepos,  # 3
                eequat,  # 4
                box_pos, # 3
                torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float),
                torch.ones((self.num_envs, 11), device=self.device, dtype=torch.float)
            ],
            axis=-1,
        )

        obs = self.obs_buf.cpu().numpy()

        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
