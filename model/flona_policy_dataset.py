"""
a train data sample is like:
an RGB image:
a shortest path: 20 * 2, consits of future 20 steps in meters in local coordinate system;
a collision-free action: 20 * 2, consits of future 20 steps in meters in local coordinate system;
"""
import os

import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset

class FlonaPolicyDataset(Dataset):
    def __init__(
        self, 
        data_folder, 
        scene_names,
        context_size,
        waypoint_spacing, 
        end_slack,   
        len_traj_pred, 
        split, 
        transform=None
        ):
        self.data_folder = data_folder
        self.scene_names = scene_names
        self.context_size = context_size
        self.waypoint_spacing = waypoint_spacing
        self.end_slack = end_slack  
        self.len_traj_pred = len_traj_pred
        self.traj_names = {}
        for scene_name in scene_names:
            self.traj_names[scene_name] = []
            for file in os.listdir(os.path.join(data_folder, scene_name)):
                if os.path.isdir(os.path.join(data_folder, scene_name, file)) and file.startswith("traj"):
                    self.traj_names[scene_name].append(file)
            self.traj_names[scene_name].sort()
        self._load_index()

    def _load_index(self):
        index_path = os.path.join(self.data_dir, 'xxx.npz') #to be modified
        try:
            index = np.load(index_path)
            self.data = index['data']
        except:
            print('Building index...')
            self.data = self._build_index()
            np.savez(index_path, data=self.data)
            print('Index built.')   
            
    def _build_caches(self):
        pass
    
    def _get_trajectory(self, scene_name, traj_name):
        pass
    
    def _build_index(self, use_tqdm: bool = False):
        
        for scene_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            trajs = self.traj_names[scene_name]
            for traj_name in trajs:
                scene_name_traj_name = scene_name + '-' + traj_name
                traj_data = self._get_trajectory(scene_name, traj_name)
                traj_len = traj_data.shape[0]
                begin_time = self.context_size * self.waypoint_spacing
                end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
                for curr_time in range(begin_time, end_time):    # is the whole seq len is less than pred_len begin_time will be greater than end_time and the loop will not run               
                    goal_min_end_time = min(traj_len - self.end_slack, curr_time + self.len_traj_pred * self.waypoint_spacing)
                    goal_max_end_time = traj_len - self.end_slack
                    for goal_time in range(goal_min_end_time, goal_max_end_time + 1):
                        samples_index.append((scene_name_traj_name, curr_time, goal_time))
    
    def _load_data(self):
        data = []
        with open(os.path.join(self.data_dir, self.split + '.txt'), 'r') as f:
            for line in f:
                data.append(line.strip())
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        return:
        RGB image,
        global shortest path,
        collision-free actions,
        (current pose)
        """
        img_path = os.path.join(self.data_dir, self.data[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img