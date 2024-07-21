import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training



def get_data_path(data_folder: str, f: str, name):
    if type(name) == int:
        return os.path.join(data_folder, f, "{:05d}.png".format(name))
    elif type(name) == str:
        return os.path.join(data_folder, f, name + ".png") 
    else:
        raise ValueError("name should be either int or str") 


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(            # 以当前点为坐标原点，当前点方向为x正方向，计算其他点的坐标
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    if len(curr_yaw) == 2:
        dir = curr_yaw - curr_pos
        #conver vector to yaw
        curr_yaw = np.arctan2(dir[1], dir[0])
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def to_global_coords(            # 以当前点为坐标原点，当前点方向为x正方向，计算其他点的坐标
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert     in local coordinates  (b,2) 
        curr_pos (np.ndarray): current position         in global coordinates  (2,)
        curr_yaw (float): current yaw                  in global coordinates   (2,)
    Returns:
        np.ndarray: positions in local coordinates
    """
    if len(curr_yaw) == 2:
        dir = curr_yaw - curr_pos
        #conver vector to yaw
        curr_yaw = np.arctan2(dir[1], dir[0])
    rotmat = yaw_rotmat(curr_yaw)  # R from local to global
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return curr_pos + positions.dot(rotmat.transpose())


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)    

def img_path_to_data_and_point_transfer(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int], cur_pos: np.ndarray, goal_pos: np.ndarray, cur_ori: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an image and two points, transform the image and transfer the points to local coordinates
    Args:
        path (str): path to the image
        cur_pos (np.ndarray): current position in the pixel coordinate of original image [x,y]
        goal_pos (np.ndarray): goal position in the pixel coordinate of original image
        image_resize_size (Tuple[int, int]): size to resize the image to [width, height]
    Returns:
        Tuple[torch.Tensor, np.ndarray, np.ndarray]: resized image as tensor, current position in the transformed image size coordinate, goal position in the same coordinate
    """
    with Image.open(path) as img:
        w, h = img.size
        x = float(cur_pos[0]) / 0.01 + w / 2
        y = float(cur_pos[1]) / 0.01 + h / 2
        cur_pos_in_oriSize = np.array([x, y])
        x = float(goal_pos[0]) / 0.01 + w / 2
        y = float(goal_pos[1]) / 0.01 + h / 2
        goal_pos_in_oriSize = np.array([x, y])
        x = float(cur_ori[0]) / 0.01 + w / 2
        y = float(cur_ori[1]) / 0.01 + h / 2
        cur_ori_in_oriSize = np.array([x, y])
        aspect_ratio = IMAGE_ASPECT_RATIO
        
        if w > h:
            img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
            w1, h1 = img.size
            cur_pos_in_cropSize = cur_pos_in_oriSize.copy()
            cur_pos_in_cropSize[0] = (w1 - w) // 2 + cur_pos_in_oriSize[0]
            goal_pos_in_cropSize = goal_pos_in_oriSize.copy()
            goal_pos_in_cropSize[0] = (w1 - w) // 2 + goal_pos_in_oriSize[0]
            cur_ori_in_cropSize = cur_ori_in_oriSize.copy()
            cur_ori_in_cropSize[0] = (w1 - w) // 2 + cur_ori_in_oriSize[0]
        else:
            img = TF.center_crop(img, (int(w / aspect_ratio), w))
            w1, h1 = img.size
            cur_pos_in_cropSize = cur_pos_in_oriSize.copy()
            cur_pos_in_cropSize[1] = (h1 - h) // 2 + cur_pos_in_oriSize[1]
            goal_pos_in_cropSize = goal_pos_in_oriSize.copy()
            goal_pos_in_cropSize[1] = (h1 - h) // 2 + goal_pos_in_oriSize[1]
            cur_ori_in_cropSize = cur_ori_in_oriSize.copy()
            cur_ori_in_cropSize[1] = (h1 - h) // 2 + cur_ori_in_oriSize[1]
            
            
        img = img.resize(image_resize_size)
        cur_pos_in_resizeSize = cur_pos_in_cropSize.copy()
        cur_pos_in_resizeSize[0] = cur_pos_in_cropSize[0] * image_resize_size[0] / w1
        cur_pos_in_resizeSize[1] = cur_pos_in_cropSize[1] * image_resize_size[1] / h1
        goal_pos_in_resizeSize = goal_pos_in_cropSize.copy()
        goal_pos_in_resizeSize[0] = goal_pos_in_cropSize[0] * image_resize_size[0] / w1
        goal_pos_in_resizeSize[1] = goal_pos_in_cropSize[1] * image_resize_size[1] / h1
        cur_ori_in_resizeSize = cur_ori_in_cropSize.copy()
        cur_ori_in_resizeSize[0] = cur_ori_in_cropSize[0] * image_resize_size[0] / w1
        cur_ori_in_resizeSize[1] = cur_ori_in_cropSize[1] * image_resize_size[1] / h1
        resize_img = TF.to_tensor(img)
        return (resize_img, cur_pos_in_resizeSize.astype(int), goal_pos_in_resizeSize.astype(int), cur_ori_in_resizeSize.astype(int))

