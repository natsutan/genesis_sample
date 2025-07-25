from typing import Dict, Sequence, Union, Optional
#from gym import spaces
from diffusion_policy.env.ur5.ur5_env import ur5Env
from diffusion_policy.env.ur5.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np





class ur5KeypointsEnv(ur5Env):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__()

        # ws = self.window_size
        #
        # if local_keypoint_map is None:
        #     # create default keypoint definition
        #     kp_kwargs = self.genenerate_keypoint_manager_params()
        #     local_keypoint_map = kp_kwargs['local_keypoint_map']
        #     color_map = kp_kwargs['color_map']
        #
        # # create observation spaces
        # Dblockkps = np.prod(local_keypoint_map['block'].shape)
        # Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        # Dagentpos = 2
        #
        # Do = Dblockkps
        # if agent_keypoints:
        #     # blockkp + agnet_pos
        #     Do += Dagentkps
        # else:
        #     # blockkp + agnet_kp
        #     Do += Dagentpos
        # # obs + obs_mask
        # Dobs = Do * 2
        #
        # low = np.zeros((Dobs,), dtype=np.float64)
        # high = np.full_like(low, ws)
        # # mask range 0-1
        # high[Do:] = 1.
        #
        # # (block_kps+agent_kps, xy+confidence)
        #
        # self.keypoint_visible_rate = keypoint_visible_rate
        # self.agent_keypoints = agent_keypoints
        # self.draw_keypoints = draw_keypoints
        # self.kp_manager = PymunkKeypointManager(
        #     local_keypoint_map=local_keypoint_map,
        #     color_map=color_map)
        # self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = ur5Env()
        kp_manager = PymunkKeypointManager.create_from_ur5_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):

        return super()._get_obs()
    
    
    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img
