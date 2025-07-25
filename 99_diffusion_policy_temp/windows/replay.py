import genesis as gs
import math
import torch

import collections
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from Genesis.genesis import xyz_to_quat


genesis_cfg = {
    "urdf" : "C:/home/myproj/genesis/UR5/ur5/ur5_robotiq85.urdf",
    "num_actions": 6,
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


CSV_FILE = 'C:/home/myproj/genesis/UR5/ur5/data/ur5_log.csv'


def del_line(l:str):
    # lから、[]を全て取り除く
    return l.replace('[', '').replace(']', '')


#                 logs = str(self.actions[i].detach().cpu().numpy().tolist()
#                             + qpos_all.detach().cpu().numpy().tolist()
#                             + box_pos.detach().cpu().numpy().tolist()) + "\n"
def read_csv():
    import csv

    qpos_list = []
    box_pos_list = []
    action_list = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            # line全てにdel_lineを適用する。
            line = list(map(lambda x: del_line(x), line))

            action_0 = float(line[0])
            action_1 = float(line[1])
            action_2 = float(line[2])
            action_3 = float(line[3])
            action_4 = float(line[4])
            action_5 = float(line[5])
            action_6 = float(line[6])
            action_7 = float(line[7]) #gripper
            action_list.append([action_0, action_1, action_2, action_3, action_4, action_5, action_6, action_7])

            qpos_0 = float(line[8])
            qpos_1 = float(line[9])
            qpos_2 = float(line[10])
            qpos_3 = float(line[11])
            qpos_4 = float(line[12])
            qpos_5 = float(line[13])
            qpos_6 = float(line[14])
            qpos_7 = float(line[15])
            qpos_8 = float(line[16])
            qpos_9 = float(line[17])
            qpos_10 = float(line[18])
            qpos_11 = float(line[19])
            qpos_list.append(
                [qpos_0, qpos_1, qpos_2, qpos_3, qpos_4, qpos_5, qpos_6, qpos_7, qpos_8, qpos_9, qpos_10, qpos_11]
            )


            box_pos_x = float(line[20])
            box_pos_y = float(line[21])
            box_pos_z = float(line[22])

            box_pos_list.append([box_pos_x, box_pos_y, box_pos_z])

    return qpos_list, box_pos_list, action_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_genesis():
    gs.init(logging_level="warning")
    num_envs = 1

    env_cfg = genesis_cfg
    dt = 0.02
    # create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(0.5 / dt),
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0, 0, 0.5),
            camera_fov=30,

        ),
        # vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=True,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=env_cfg["base_box_pos"],
            fixed=True,
        )
    )

    # add robot
    file = genesis_cfg["urdf"]
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=file,
            fixed=True,
        ),
    )
    # build
    scene.build()

    # names to indices
    motors_dof_idx = list(np.arange(6))
    all_dof_idx = list(np.arange(12))

    # PD control parameters
    robot.set_dofs_kp(env_cfg["kp"], all_dof_idx)
    robot.set_dofs_kv(env_cfg["kd"], all_dof_idx)
    robot.set_dofs_force_range(
        env_cfg["force_limit_l"],
        env_cfg["force_limit_u"]
    )
    qpos = env_cfg["base_init_pos"]
    # qpos[0:12]を [num_envs, 12] の形にコピーしながら変形する。
    qpos = torch.tensor(qpos, device=device, dtype=torch.float)
    robot.set_qpos(qpos)

    qpos_list, box_pos_list, action_list = read_csv()
    box_pos = torch.tensor(box_pos_list[0], device=device, dtype=torch.float)
    cube.set_pos(box_pos)

    scene.step()

    for qpos, box_pos, action in zip(qpos_list, box_pos_list, action_list):
        box_pos = torch.tensor(box_pos_list[0], device=device, dtype=torch.float)
        cube.set_pos(box_pos)

        actions_t = torch.tensor(action, device=device, dtype=gs.tc_float)
        actions = torch.clip(actions_t, -env_cfg["clip_actions"], env_cfg["clip_actions"])

        qpos_all = robot.get_dofs_position(all_dof_idx)
        # only use the first 6 dofs for control
        pos, quat = robot.forward_kinematics(qpos_all)
        eepos = pos[:, 6]  # shape: (num_envs, 3)
        eequat = quat[:, 6]  # shape: (num_envs, 4)

        # eequatをdeg(rx, ry, rz)に変換
        ee_deg = quat_to_xyz(eequat)

        delta_pos = actions * env_cfg["action_scale"]
        target_eepos = eepos + delta_pos[0:3]
        target_eedeq = ee_deg + delta_pos[3:6] * env_cfg["action_scale_deg"]

        target_eequat = xyz_to_quat(target_eedeq)

        target_dof_pos = robot.inverse_kinematics(
            link=robot.get_link("wrist_3_link"),
            pos=target_eepos,
            quat=target_eequat,
            respect_joint_limit=True,
            dofs_idx_local=motors_dof_idx,
        )

        robot.set_qpos(target_dof_pos[:, 0:6], motors_dof_idx)
        robot.zero_all_dofs_velocity()

        scene.step()






def main():
    run_genesis()


if __name__ == '__main__':
    main()