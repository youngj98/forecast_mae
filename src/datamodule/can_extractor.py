import os
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.optim as optim
import scipy.io as sio
from scipy.interpolate import interp1d

class CANExtractor:
    def __init__(
        self,
        radius: float = 150,
        mode: str = "train",
        remove_outlier_actors: bool = True,
    ) -> None:
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors

    def save(self, file: str):
        pass
    
    def get_data(self, raw_data, index):
        return self.process(raw_data, index)
    
    def process(self, raw_data, index):
        can_yaw_rate = torch.tensor(raw_data['can_yaw_rate'], dtype=torch.float)
        can_wheel_speed = torch.tensor(raw_data['can_wheel_speed'], dtype=torch.float)
        can_steering_spd = torch.tensor(raw_data['can_steering_spd'], dtype=torch.float)
        can_steering_ang = torch.tensor(raw_data['can_steering_ang'], dtype=torch.float)
        can_lateral_accel = torch.tensor(raw_data['can_lateral_accel'], dtype=torch.float)
        can_longitudinal_accel = torch.tensor(raw_data['can_longitudinal_accel'], dtype=torch.float)
        
        dr_x = torch.tensor(raw_data['dr_x'], dtype=torch.float)
        dr_y = torch.tensor(raw_data['dr_y'], dtype=torch.float)
        y = torch.stack((dr_x, dr_y), dim=-1)
        
        padding_mask = (y == 0).all(dim=-1)
        padding_mask[0, -1] = False
        
        origin = torch.tensor([0, 0], dtype=torch.float)
        theta = torch.tensor([0], dtype=torch.float)
        scenario_id = torch.tensor([0], dtype=torch.int) 
        agent_id = torch.tensor([0], dtype=torch.int) 
        city = torch.tensor([0], dtype=torch.int) 
        
        return {
            "y": y,
            "can_yaw_rate": can_yaw_rate,
            "can_wheel_speed": can_wheel_speed,
            "can_steering_spd": can_steering_spd,
            "can_steering_ang": can_steering_ang,
            "can_lateral_accel": can_lateral_accel,
            "can_longitudinal_accel": can_longitudinal_accel,
            "padding_mask": padding_mask,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
        }
        pass