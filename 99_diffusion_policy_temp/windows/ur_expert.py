import os
#os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import argparse

from ur_bc_env import URBCEnv


env_cfg = {
    "num_actions": 6,  # Δx、Δθ
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
    "base_init_quat": [-0.5,  0.5, -0.5,  0.5],

    "base_box_pos": [0.65, 0.0, 0.02],
    "base_box_quat": [1.0, 0, 0, 0],  # [x, y, z, w]
    "box_pos_randamin_range": 0.20,  # box position random range
}

def main():
    print("Genesis version:", gs.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="ur-pick")
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    env_cfg["exp_name"] = args.exp_name

    env = URBCEnv(
        env_cfg=env_cfg,   show_viewer=False
    )

    for i in range(args.max_iterations):
        print(f"Iteration {i+1}/{args.max_iterations}")
        env.reset()
        env.run()




if __name__ == "__main__":
    main()



