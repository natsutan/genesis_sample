import zarr
import glob
import json
import os
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

INPUT_DIR = '/home/natu/myproj/diffusion_policy/data/ur5_pick/20250807_224414/'
OUTPUT_DIR = '/home/natu/myproj/diffusion_policy/data/ur5_pick/'

def read_data(json_file):
    actions = []
    states = []

    with open(json_file, 'r') as f:
        # 一行ずつ読む
        j = json.load(f)
        cnt = 0
        for data in j['data']:
            # float型のリストに変換

            # eepos, eequat, box_pos, self.girpper_on_prev, eepos_prime, eequat_prime, self.gripper_on
            eepos = data["eepos"]
            eequat = data["eequat"]
            # 実験
            box_pos = data["box_pos"]

            is_gripper_closed = data["is_gripper_closed"]

            eepos_prime = data["eepos_prime"]
            eequat_prime = data["eequat_prime"]
            is_gripper_closed_prime = data["is_gripper_closed_prime"]

            eexyz = quat_to_xyz(np.array(eequat))
            eexyz_prime = quat_to_xyz (np.array(eequat_prime))

            state = eepos + eexyz.tolist() + box_pos + [is_gripper_closed]
            dx = [eepos_prime[0] - eepos[0], eepos_prime[1] - eepos[1], eepos_prime[2] - eepos[2]]
            dt = [eexyz_prime[0] - eexyz[0], eexyz_prime[1] - eexyz[1], eexyz_prime[2] - eexyz[2]]
            dg = is_gripper_closed_prime - is_gripper_closed

            action = dx + dt + [dg]
            states.append(state)
            actions.append(action)

            cnt += 1
            if is_gripper_closed_prime > 0.5:
                break


        episode_ends = cnt


    return actions, states, episode_ends



def read_all_data(csv_dir):
    all_action = []
    all_state = []
    all_episode_end = []

    csvs = glob.glob(os.path.join(csv_dir, '*.json'))
    print(csvs)

    last = 0
    for csv_file in csvs:
        action, state, episode_end = read_data(csv_file)
        all_action.extend(action)
        all_state.extend(state)
        last = last + episode_end
        all_episode_end.extend([last])

    return all_action, all_state, all_episode_end

def create_zarr(actions, states, episode_ends, output_dir):
    output_file = os.path.join(output_dir, "ur5_pick.zarr")
    store = zarr.DirectoryStore(output_file)
    root = zarr.group(store=store, overwrite=True)

    z_actions = zarr.array(actions, chunks=(1, len(actions[0])), dtype='float32')
    z_states = zarr.array(states, chunks=(1, len(states[0])), dtype='float32')

    root.create_dataset('meta/episode_ends', data=episode_ends)
    root.create_dataset('data/state', data=z_states)
    root.create_dataset('data/action', data=z_actions)

    print(root.info)
    print('saved', output_file)

if __name__ == '__main__':
    action, state, episode_end = read_all_data(INPUT_DIR)
    create_zarr(action, state, episode_end, OUTPUT_DIR)