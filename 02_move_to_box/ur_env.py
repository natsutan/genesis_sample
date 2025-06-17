import torch
import math
import socket
import json
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from taichi.examples.simulation.mass_spring_game import fixed

from Genesis.genesis import xyz_to_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# WSL2 マシン(またはサーバー側)のIPアドレスとポートを合わせる
# >wsl -- ip addr で確認できる
HOST = '172.27.55.153'  # localhostで試す場合、WSL2/Windows間の通信は別IPになることも
PORT = 50009

def ompl_waypoints(start, goal, num_waypoint):
    start_list = [float(x) for x in start]
    goal_list = [float(x) for x in goal]

    # 送信したいデータ
    data_to_send = {
        "qpos_start": start_list,
        "qpos_goal":  goal_list,
        "num_waypoint": num_waypoint
    }

    # ソケットを作ってサーバーに接続
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to server {HOST}:{PORT} ...")
        s.connect((HOST, PORT))

        # JSONでエンコードして送信
        message = json.dumps(data_to_send).encode()
        s.sendall(message)

        # レスポンス受信

        # JSONデコードして結果を表示
        response_data = recv_all(s)
        response = json.loads(response_data.decode())
        print("Received:", response)
        return response.get("waypoint")


def recv_all(sock):
    """ サーバーが送信を完了 or ソケット閉じるまで、繰り返し受信する """
    buffers = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            # サーバー側が close した
            break
        buffers.append(chunk)
    return b"".join(buffers)


class UREnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
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
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
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
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="D:/home/myproj/genesis/UR5/asset/ur5/ur5_robotiq85.urdf",
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
        qpos = torch.tensor(qpos, device=gs.device, dtype=gs.tc_float).repeat(num_envs, 1)
        self.robot.set_qpos(qpos, envs_idx=list(range(num_envs)))

        self.scene.step()

        box_pos = self.cube.get_dofs_position()
        print("box_pos", box_pos)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        pass

    def step(self, actions):
        # print("actions", actions[0])
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        qpos_all = self.robot.get_dofs_position(self.all_dof_idx) # only use the first 6 dofs for control
        pos, quat = self.robot.forward_kinematics(qpos_all)
        eepos = pos[:, 6]  # shape: (num_envs, 3)
        eequat = quat[:, 6]  # shape: (num_envs, 4)

        # eequatをdeg(rx, ry, rz)に変換
        ee_deg = quat_to_xyz(eequat)

        delta_pos = self.actions * self.env_cfg["action_scale"]
        target_eepos = eepos + delta_pos[:, 0:3]
        target_eedeq = ee_deg + delta_pos[:, 3:6] *  self.env_cfg["action_scale_deg"]

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

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.reached_goal

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

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
                self.actions,  # 6
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos[0:6]
        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        num_envs = self.num_envs
        qpos = torch.tensor(qpos, device=gs.device, dtype=gs.tc_float).repeat(len(envs_idx), 1)
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

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_ee_pos(self):
        reward, pos_error = self.calc_eepos_reward()

        pos_error_ok = pos_error < self.reward_cfg["pos_error_threshold"]
        reward[pos_error_ok] += self.reward_cfg["pos_bonus"]

        return torch.tensor(reward, device=gs.device, dtype=gs.tc_float)

    def _reward_ee_quat(self):

        reward, angle = self.calc_eequat_reward()

        angle_threshold_deg = self.reward_cfg["quat_angle_threshold_deg"]
        angle_threshold_rad = torch.tensor(angle_threshold_deg * np.pi / 180.0, device=gs.device)

        # しきい値以下なら追加報酬
        mask = angle < angle_threshold_rad
        reward[mask] += self.reward_cfg["quat_bonus"]

        return reward

    def _reward_ee_both(self):

        # --- 位置誤差（L2ノルムの2乗） ---
        _ , pos_error = self.calc_eepos_reward()
        pos_threshold = self.reward_cfg["pos_error_threshold"]

        # --- クオータニオン角度誤差 ---
        _, angle = self.calc_eequat_reward()

        quat_threshold_deg = self.reward_cfg["quat_angle_threshold_deg"]
        quat_threshold_rad = quat_threshold_deg * np.pi / 180.0

        # --- 両方OKなら報酬を与える ---
        pos_ok = pos_error < pos_threshold
        quat_ok = angle < quat_threshold_rad
        both_ok = pos_ok & quat_ok

        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        reward[both_ok] += self.reward_cfg["both_bonus"]

        return reward

    def _reward_ee_x(self):
        # xが負の値の時はペナルティを与えエピソードを終了する。
        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        pos, quat = self.robot.forward_kinematics(qpos)
        eepos = pos[:,6].cpu().numpy()

        pos_x_err = eepos[:, 0] < 0.0
        self.reached_goal[pos_x_err] = True
        self.rew_buf[pos_x_err] = self.reward_cfg["bad_pose_penalty"]
        reward = np.zeros(self.num_envs, dtype=np.float32)
        reward[pos_x_err] = self.reward_cfg["bad_pose_penalty"]

        return torch.tensor(reward, device=gs.device, dtype=gs.tc_float)

    def _reward_ee_quat_max(self):
        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        _, quat = self.robot.forward_kinematics(qpos)
        eequat = quat[:, 6]  # shape: (num_envs, 4)

        # 目標クオータニオン
        target_quat = torch.tensor(
            self.reward_cfg["target_quat"], device=gs.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)

        # 回転角度 θ = 2 * arccos(|dot|)
        dot = torch.sum(eequat * target_quat, dim=1).clamp(-1.0, 1.0)
        angle = 2.0 * torch.arccos(torch.abs(dot))  # ラジアン

        # 最大許容角度（度 → ラジアン）
        max_angle_deg = self.reward_cfg["quat_max_angle_deg"]
        max_angle_rad = torch.tensor(max_angle_deg * np.pi / 180.0, device=gs.device)

        # マスク：許容範囲を超えた環境
        quat_err = angle > max_angle_rad
        #print("angle", angle.cpu().numpy(), "max_angle_rad", max_angle_rad.cpu().numpy(), "quat_err", quat_err.cpu().numpy())

        # エピソード終了フラグと大きな罰
        self.reached_goal[quat_err] = True
        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        reward[quat_err] = self.reward_cfg["bad_pose_penalty"]

        return reward

    def _reward_collision(self):
        # 衝突があった場合、報酬を-100.0に設定し、エピソードを終了する。
        contacts = self.robot.get_contacts(with_entity=self.plane)

        try:
            collision_idx = contacts["geom_a"][:,0] == 0
        except IndexError:
            # 衝突がない場合、collision_idxは全てfalseになる
            collision_idx = torch.tensor([False] * self.num_envs)
        # print("collision_idx", collision_idx.cpu().numpy())

        self.reached_goal[collision_idx] = True
        reward = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        reward[collision_idx] = -1.0

        return reward


    def calc_eepos_reward(self):
        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        pos, quat = self.robot.forward_kinematics(qpos)
        eepos = pos[:,6]
        target_pos = torch.tensor(self.reward_cfg["target_pos"], device=gs.device, dtype=gs.tc_float)
        box_pos = self.cube.get_pos()
        offset = torch.tensor([0.0, 0.0, 0.02], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        target_pos = box_pos + offset
        #
        # target_pos_debug = target_pos[0].cpu().numpy()
        # print(target_pos_debug)

        # Calculate the distance between the end effector position and the target position
        pos_error = torch.pow(eepos - target_pos, 2).sum(dim=1)
        # Calculate the orientation error using quaternion distance

        reward = 5.0 / (1.0 + pos_error)

        return reward, pos_error

    def calc_eequat_reward(self):
        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        _, quat = self.robot.forward_kinematics(qpos)
        eequat = quat[:, 6]  # shape: (num_envs, 4)

        # 目標クオータニオンをテンソル化
        target_quat = torch.tensor(
            self.reward_cfg["target_quat"], device=gs.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)  # shape: (num_envs, 4)

        # dot = cos(θ/2)
        dot = torch.sum(eequat * target_quat, dim=1).clamp(-1.0, 1.0)
        angle = 2.0 * torch.arccos(torch.abs(dot))  # ラジアン

        # 角度しきい値（ラジアン）を指定

        # 報酬：誤差が小さいほど大きく、しきい値以下でボーナス
        # reward = -angle  # 基本は負の報酬（距離の近さに対応）

        reward = 1.0 / (1.0 + angle) # 角度が小さいほど報酬が大きくなる

        return reward, angle
