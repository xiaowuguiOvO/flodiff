import os
import sys
from typing import List, Tuple

import numpy as np
import tqdm
import lmdb
import torch
from torch.utils.data import Dataset


from .data_utils import (
    img_path_to_data,
    img_path_to_data_and_point_transfer,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class flona_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        trav_folder: str,   
        scene_names: List[str],
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        len_traj_pred: int,
        context_size: int,
        end_slack: int = 0,
        normalize: bool = True,
    ):
        self.data_folder = data_folder
        self.scene_names = scene_names
        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.context_size = context_size
        self.end_slack = end_slack
        self.normalize = normalize
        self.metric_waypoint_spacing = 0.045 #0.255    # 0.045
        self._image_caches = {}
        self.traj_names = {}    # {scene_name: [traj_name]}
        for scene_name in scene_names:
            self.traj_names[scene_name] = []
            for file in os.listdir(os.path.join(data_folder, scene_name)):
                if os.path.isdir(os.path.join(data_folder, scene_name, file)) and file.startswith("traj"):
                    self.traj_names[scene_name].append(file)
            self.traj_names[scene_name].sort()
        print('traj_names produced and size ', sys.getsizeof(self.traj_names))
        self.trajectory_cache = {} # {scene_name: {traj_name: traj_data}}
        self._load_index()
        self.floor_shapes_ori =  np.load(os.path.join(trav_folder, "floor_shapes.npy"), allow_pickle=True).item()

        self.num_action_params = 2
        
    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        print("Building LMDB cache")
        self._image_caches = {}
        
        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for scene_name in self.scene_names:
            for traj_name in self.traj_names[scene_name]:
                self._get_trajectory(scene_name, traj_name)
        print('trajectories loaded and size ', sys.getsizeof(self.trajectory_cache))
        
        for scene_name in self.scene_names:
            cache_filename = os.path.join(self.data_folder, scene_name, f"dataset_{scene_name}.lmdb")
            if not os.path.exists(cache_filename):
                start_idx = self.scene_start_index[self.scene_names.index(scene_name)]
                end_idx = self.scene_end_index[self.scene_names.index(scene_name)]
                base_idx = start_idx
                tqdm_iterator = tqdm.tqdm(
                    self.index_to_data_str[start_idx:end_idx + 1],
                    disable=not use_tqdm,
                    dynamic_ncols=True,
                    desc=f"Building LMDB cache for {scene_name}",
                )
                with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                    with image_cache.begin(write=True) as txn:
                        for idx, scene_name_and_traj_name in enumerate(tqdm_iterator):
                            sn, traj_name = scene_name_and_traj_name.split('-')
                            time = self.index_to_data_int[idx + base_idx, 0]
                            image_path = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, time)
                            with open(image_path, "rb") as f:
                                txn.put(image_path.encode(), f.read())
                            
                        trajs = self.traj_names[scene_name]
                        for traj_name in trajs:
                            image_path_0 = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, 0)
                            image_path_1 = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, 1) 
                            image_path_2 = get_data_path(os.path.join(self.data_folder, scene_name), traj_name, 2)
                            with open(image_path_0, "rb") as f:
                                txn.put(image_path_0.encode(), f.read())
                            with open(image_path_1, "rb") as f:
                                txn.put(image_path_1.encode(), f.read())
                            with open(image_path_2, "rb") as f:
                                txn.put(image_path_2.encode(), f.read())                      
                        image_path = get_data_path(self.data_folder, scene_name, "floorplan")
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())
            self._image_caches[scene_name] = lmdb.open(cache_filename, readonly=True)
         

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time)
        """
        samples_index = []
        goals_pos = []
        scene_start_index = []
        
        for scene_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            trajs = self.traj_names[scene_name]
            scene_start_index.append(len(samples_index))
            for traj_name in trajs:
                scene_name_traj_name = scene_name + '-' + traj_name
                traj_data = self._get_trajectory(scene_name, traj_name)
                traj_len = traj_data.shape[0]
                goal_pos = traj_data[-1 - self.end_slack,:2]
                goals_pos.append(goal_pos)
                begin_time = self.context_size * self.waypoint_spacing
                end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
                for curr_time in range(begin_time, end_time):    
                    goal_min_end_time = min(traj_len - self.end_slack, curr_time + self.len_traj_pred * self.waypoint_spacing)
                    goal_max_end_time = traj_len - self.end_slack
                    for goal_time in range(goal_min_end_time, goal_max_end_time + 1):
                        samples_index.append((scene_name_traj_name, curr_time, goal_time))
        scene_end_index = []
        for i in range(len(scene_start_index) - 1):
            scene_end_index.append(scene_start_index[i + 1] - 1)
        scene_end_index.append(len(samples_index) - 1)
        return samples_index, goals_pos, scene_start_index, scene_end_index

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_folder,
            f"dataset_n{self.context_size}_slack_{self.end_slack}.npz",
        )
     
        try:
            # load the index_to_data if it already exists (to save time)
            npz = np.load(index_to_data_path, allow_pickle=True)
            # self.index_to_data = npz["index_to_data"]
            index_to_data = npz["index_to_data"]
            self.index_to_data_str = index_to_data[:,0].copy()
            self.index_to_data_int = index_to_data[:,1:].copy().astype(np.int16)
            del index_to_data
            self.goals_pos = npz["goals_pos"]
            print('Index loaded, index_to_data size', sys.getsizeof(self.index_to_data_str))

        except:
            # if the index_to_data file doesn't exist, create it
            print('Building index')
            index_to_data, self.goals_pos, self.scene_start_index, self.scene_end_index = self._build_index()
            index_to_data = np.array(index_to_data, dtype=object)
            self.goals_pos = np.array(self.goals_pos, dtype=object)
            self.index_to_data_str = index_to_data[:,0].copy()
            self.index_to_data_int = index_to_data[:,1:].copy().astype(np.int16)
            print('get index')

            np.savez(index_to_data_path, index_to_data=index_to_data, goals_pos=self.goals_pos)
            del index_to_data
            print('Index built')

    def _load_image(self, scene_name, trajectory_name, name): 
        if name == "floorplan":
            image_path = get_data_path(self.data_folder, scene_name, name)
        else:
            image_path = get_data_path(os.path.join(self.data_folder, scene_name), trajectory_name, name)
        
        try:   # directedly load from disk
            with open(image_path, "rb") as f:
                result = img_path_to_data(f, self.image_size)

            return result
            
        except TypeError:
            print(f"Failed to load image {image_path}")
            
    def _load_image_and_transform_points(self, scene_name, trajectory_name, cur_pos, goal_pos, cur_ori, name):
        cur_pos_metric = cur_pos * self.metric_waypoint_spacing * self.waypoint_spacing # trans from waypoints to meters
        goal_pos_metric = goal_pos * self.metric_waypoint_spacing * self.waypoint_spacing
        cur_ori_metric = cur_ori * self.metric_waypoint_spacing * self.waypoint_spacing
        
        if name == "floorplan":
            image_path = get_data_path(self.data_folder, scene_name, name)
        else:
            image_path = get_data_path(os.path.join(self.data_folder, scene_name), trajectory_name, name)

        try:
            with open(image_path, "rb") as f:
                result = img_path_to_data_and_point_transfer(f, self.floor_shapes_ori[scene_name], self.image_size, cur_pos_metric, goal_pos_metric, cur_ori_metric)
            return result
        except TypeError:
            print(f"Failed to load image {image_path}")
            
    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data[start_index:end_index:self.waypoint_spacing, 2:].copy()
        positions = traj_data[start_index:end_index:self.waypoint_spacing, :2].copy()
        goal_pos = traj_data[goal_time, :2].copy()
        cur_pos = positions[0]
        cur_ori = yaw[0]
        cur_ori = cur_pos + (cur_ori - cur_pos) / np.linalg.norm(cur_ori - cur_pos)
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos_local = to_local_coords(goal_pos, positions[0], yaw[0])
        cur_pos_local = to_local_coords(cur_pos, positions[0], yaw[0])
        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
        actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing  # transform meters to waypoints(steps)
            goal_pos /= self.metric_waypoint_spacing * self.waypoint_spacing
            cur_pos /= self.metric_waypoint_spacing * self.waypoint_spacing 
            cur_ori /= self.metric_waypoint_spacing * self.waypoint_spacing
            goal_pos_local /= self.metric_waypoint_spacing * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos, cur_pos, cur_ori, goal_pos_local, cur_pos_local
    
    def _compute_actions_ori(self, traj_data, curr_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data[start_index:end_index:self.waypoint_spacing, 2:].copy()
        positions = traj_data[start_index:end_index:self.waypoint_spacing, :2].copy()
        goal_pos = traj_data[-1 - self.end_slack, :2].copy()
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        cur_pos = waypoints[0]
       
        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        actions = waypoints[1:]
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing  # transform meters to waypoints(steps)
            goal_pos /= self.metric_waypoint_spacing * self.waypoint_spacing
            cur_pos /= self.metric_waypoint_spacing * self.waypoint_spacing 

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos, cur_pos
    
    def _get_trajectory(self, scene_name, trajectory_name):
        k = scene_name+'-'+trajectory_name
        if k in self.trajectory_cache:
            return self.trajectory_cache[k]
        else:
            with open(os.path.join(self.data_folder, scene_name, trajectory_name, trajectory_name + ".npy"), "rb") as f:
                traj_data = np.load(f)
            self.trajectory_cache[k] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data_str)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        scene_name_cur_and_f_curr = self.index_to_data_str[i]
        curr_time, goal_time = self.index_to_data_int[i]
        scene_name_cur, f_curr = scene_name_cur_and_f_curr.split('-')
        # Load images
        context = []
        # sample the last self.context_size times from interval [0, curr_time)
        context_times = list(
            range(
                curr_time + -self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )
        context = [(f_curr, t) for t in context_times]

        obs_image = torch.cat([
            self._load_image(scene_name_cur, f, t) for f, t in context
        ])
        # Load actions and current position and goal position
        curr_traj_data = self._get_trajectory(scene_name_cur, f_curr) # meters metric
        curr_traj_len =  curr_traj_data.shape[0]
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"
        actions, goal_pos, cur_pos, cur_ori, goal_pos_local, cur_pos_local = self._compute_actions(curr_traj_data, curr_time, goal_time)   # waypoints(steps) metric in local        
        # Load goal image
        floorplan_image, cur_pos_resized, goal_pos_resized, cur_ori_resized  = self._load_image_and_transform_points(scene_name_cur, f_curr, cur_pos, goal_pos, cur_ori, "floorplan")

        # Compute distances
        # distance = (len(curr_traj_data) - self.end_slack - curr_time) // self.waypoint_spacing
        distance = (goal_time - curr_time) // self.waypoint_spacing
        assert (len(curr_traj_data) - self.end_slack - curr_time) % self.waypoint_spacing == 0, f"{len(curr_traj_data) - self.end_slack} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"      
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),   # [3*L, H, W]
            torch.as_tensor(floorplan_image, dtype=torch.float32),
            actions_torch,           # waypoints(steps) metric in local
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(cur_pos, dtype=torch.float32),
            torch.as_tensor(cur_ori, dtype=torch.float32),
            torch.as_tensor(goal_pos_resized, dtype=torch.float32),
            torch.as_tensor(cur_pos_resized, dtype=torch.float32),
            torch.as_tensor(cur_ori_resized, dtype=torch.float32),
            torch.as_tensor(goal_pos_local, dtype=torch.float32),
            torch.as_tensor(cur_pos_local, dtype=torch.float32)
        )


