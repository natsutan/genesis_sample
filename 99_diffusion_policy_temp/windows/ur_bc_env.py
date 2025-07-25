import os
import torch
import socket
import json
import datetime
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

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


class URBCEnv:
    def __init__(self, env_cfg, show_viewer=False):
        self.device = gs.device

        self.dt = 0.01  # control frequency on real robot is 100hz

        self.env_cfg = env_cfg

        # self.obs_scales = obs_cfg["obs_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0, 0, 0.5),
                camera_fov=30,

            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                box_box_detection=True,
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
                #fixed=True,
            )
        )

        # add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="D:/home/myproj/genesis/UR5/asset/ur5/ur5_robotiq85.urdf",
                fixed=True,
            ),
        )

        # logsの下に時刻を文字列にしたディレクトリを作る。
        self.log_dir = f"logs/{self.env_cfg['exp_name']}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.logn = 0
        self.logf = open(f"{self.log_dir}/log{self.logn:03d}.csv", "w")

        # build
        self.scene.build()

        # names to indices
        self.motors_dof_idx = np.arange(6)
        self.fingers_dof_idx = np.arange(6, 12)
        self.all_dof_idx = np.arange(12)
        self.end_effector = self.robot.get_link('wrist_3_link')

        # PD control parameters
        self.robot.set_dofs_kp(self.env_cfg["kp"], self.all_dof_idx)
        self.robot.set_dofs_kv(self.env_cfg["kd"], self.all_dof_idx)
        self.robot.set_dofs_force_range(
            self.env_cfg["force_limit_l"],
            self.env_cfg["force_limit_u"]
        )
        self.finger_qpos = [0.00, ]
        self.gripper_on = 0
        self.gripper_on_prev = 0

        self.default_dof_pos = self.env_cfg["base_init_pos"]
        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        qpos = torch.tensor(qpos, device=gs.device, dtype=gs.tc_float)
        self.robot.set_qpos(qpos)
        self.first_quat = self.env_cfg["base_init_quat"]

        self.link_cube = np.array([self.cube.get_link("box_baselink").idx], dtype=gs.np_int)
        self.link_ee_l = np.array([self.robot.get_link("robotiq_85_left_finger_tip_link").idx], dtype=gs.np_int)
        self.link_ee_r = np.array([self.robot.get_link("robotiq_85_right_finger_tip_link").idx], dtype=gs.np_int)

        self.scene.step()



    def run(self):

        self.gripper_open()
        qpos = self.robot.get_qpos()
        for i in range(30):
            self.robot.control_dofs_position(qpos[0:6], self.motors_dof_idx)
            self.robot.control_dofs_position(self.finger_qpos, self.fingers_dof_idx)
            self.log(qpos)
            self.scene.step()

        # workの真上に移動
        start_qpos = self.robot.get_qpos()
        box_pos = self.cube.get_pos()
        box_pos += torch.tensor([0.0, 0.0, 0.4])  # z方向に0.4m上昇
        goal_qpos = self.robot.inverse_kinematics(
            link=self.end_effector,
            pos=box_pos,
            quat = self.first_quat,
            respect_joint_limit = True,
            dofs_idx_local = self.motors_dof_idx
        )

        waypoints  = ompl_waypoints(start_qpos, goal_qpos, 100)
        for qpos in waypoints:
            self.robot.control_dofs_position(qpos[0:6], self.motors_dof_idx)
            self.robot.control_dofs_position(self.finger_qpos, self.fingers_dof_idx)
            self.log(qpos)
            self.scene.step()


        start_qpos = self.robot.get_qpos()
        box_pos = self.cube.get_pos()
        box_pos[2] = 0.25
        goal_qpos = self.robot.inverse_kinematics(
            link=self.end_effector,
            pos=box_pos,
            quat = self.first_quat,
            respect_joint_limit = True,
            dofs_idx_local = self.motors_dof_idx
        )

        waypoints  = ompl_waypoints(start_qpos, goal_qpos, 100)
        for qpos in waypoints:
            self.robot.control_dofs_position(qpos[0:6], self.motors_dof_idx)
            self.robot.control_dofs_position(self.finger_qpos, self.fingers_dof_idx)
            self.log(qpos)
            self.scene.step()

        self.gripper_close()
        qpos = self.robot.get_qpos()
        for i in range(30):
            self.robot.control_dofs_position(qpos[0:6], self.motors_dof_idx)
            self.robot.control_dofs_position(self.finger_qpos, self.fingers_dof_idx)
            self.log(qpos)
            self.scene.step()
            contacts = self.robot.get_contacts(with_entity=self.cube)
            link_b = contacts["link_b"]
            if link_b.shape[0] > 0:
                link_b = link_b.cpu().tolist()
                if self.link_ee_l in link_b and self.link_ee_r in link_b:  # 18 is the index for the gripper fingers
                    self.scene.sim.rigid_solver.add_weld_constraint(self.link_ee_l, self.link_cube)
                    # self.scene.sim.rigid_solver.add_weld_constraint(self.link_ee_r, self.link_cube)
                    break

        # self.gripper_open()
        # qpos = self.robot.get_qpos()
        # for i in range(50):
        #     self.robot.control_dofs_position(qpos[0:6], self.motors_dof_idx)
        #     self.robot.control_dofs_position(self.finger_qpos, self.fingers_dof_idx)
        #     self.scene.step()


        # 元に戻す
        start_qpos = self.robot.get_qpos()
        goal_qpos = self.env_cfg["base_init_pos"]

        waypoints  = ompl_waypoints(start_qpos, goal_qpos, 100)
        for qpos in waypoints:
            self.robot.control_dofs_position(qpos[0:6], self.motors_dof_idx)
            self.robot.control_dofs_position(self.finger_qpos, self.fingers_dof_idx)
            self.log(qpos)
            self.scene.step()

        self.scene.sim.rigid_solver.delete_weld_constraint(self.link_ee_l, self.link_cube)

    def reset_idx(self):

        # log
        self.logf.close()
        self.logn += 1
        self.logf = open(f"{self.log_dir}/log{self.logn:03d}.csv", "w")

        # reset dofs
        self.dof_pos = self.default_dof_pos[0:6]
        qpos = self.env_cfg["base_init_pos"]
        # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
        qpos = torch.tensor(qpos, device=gs.device, dtype=gs.tc_float)
        self.robot.set_qpos(qpos)
        self.gripper_on = 0
        self.gripper_on_prev = 0

        box_pos = torch.tensor(self.env_cfg["base_box_pos"], device=gs.device, dtype=gs.tc_float)
        box_quat = torch.tensor(self.env_cfg["base_box_quat"], device=gs.device, dtype=gs.tc_float)

        # box_posのxとyをランダムで±0.1の範囲に変化させる
        range = self.env_cfg["box_pos_randamin_range"]
        rand_x = (range * 2 * torch.rand(1, device=gs.device) - range)
        box_pos[0] += rand_x[0]
        rand_y = (range * 2 * torch.rand(1, device=gs.device) - range)
        box_pos[1] += rand_y[0]

        self.cube.set_pos(box_pos)
        self.cube.set_quat(box_quat)

        # reset base
        self.robot.zero_all_dofs_velocity()



    def reset(self):
        self.reset_idx()

    def gripper_open(self):
        self.gripper_on_prev = self.gripper_on
        self.gripper_on = 1
        self.finger_qpos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def gripper_close(self):
        self.gripper_on_prev = self.gripper_on
        self.gripper_on = 0
        self.finger_qpos = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]


    def log(self, qpos_prime):
        qpos = self.robot.get_dofs_position(self.all_dof_idx)
        pos, quat = self.robot.forward_kinematics(qpos)
        eepos = pos[6]
        eequat = quat[6]
        box_pos = self.cube.get_pos()

        qpos_prime = torch.tensor(qpos_prime, device=gs.device, dtype=gs.tc_float)
        pos_prime, quat_prime = self.robot.forward_kinematics(qpos_prime)
        eepos_prime = pos_prime[6]
        eequat_prime = quat_prime[6]

        # eepos, eequat, box_pos, self.girpper_on_prev, eepos_prime, eequat_prime, self.gripper_onをログに書き込む
        log_data = f"{eepos[0].item()},{eepos[1].item()},{eepos[2].item()},"
        log_data += f"{eequat[0].item()},{eequat[1].item()},{eequat[2].item()},{eequat[3].item()},"
        log_data += f"{box_pos[0].item()},{box_pos[1].item()},{box_pos[2].item()},"
        log_data += f"{self.gripper_on_prev},"
        log_data += f"{eepos_prime[0].item()},{eepos_prime[1].item()},{eepos_prime[2].item()},"
        log_data += f"{eequat_prime[0].item()},{eequat_prime[1].item()},{eequat_prime[2].item()},{eequat_prime[3].item()},"
        log_data += f"{self.gripper_on}\n"
        self.logf.write(log_data)
