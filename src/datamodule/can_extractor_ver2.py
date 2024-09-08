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
    
    def process(raw_data):
        can_data_list = np.empty((len(raw_data), 110, 7), dtype=np.float32)
        dr_data_list = np.empty((len(raw_data), 110, 2), dtype=np.float32)
        can_yaw_rate_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        can_wheel_speed_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        can_steering_spd_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        can_steering_ang_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        can_lateral_accel_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        can_longitudinal_accel_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        dr_x_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        dr_y_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        

        for i in range(len(raw_data)):
            can_data = np.empty((110, 7), dtype=np.float32)
            dr_data = np.empty((110, 2), dtype=np.float32)
            can_yaw_rate = np.empty((110, 1), dtype=np.float32)
            can_wheel_speed = np.empty((110, 1), dtype=np.float32)
            can_steering_spd = np.empty((110, 1), dtype=np.float32)
            can_steering_ang = np.empty((110, 1), dtype=np.float32)
            can_lateral_accel = np.empty((110, 1), dtype=np.float32)
            can_longitudinal_accel = np.empty((110, 1), dtype=np.float32)
            dr_x = np.empty((110, 1), dtype=np.float32)
            dr_y = np.empty((110, 1), dtype=np.float32)
            
            for j in range(len(raw_data[i])):
                for k in range(len(raw_data[i][j])):
                    raw_data[i][j] = np.array(raw_data[i][j], dtype=np.float32)
                    raw_data[i][j] = torch.tensor(raw_data[i][j], dtype=torch.float)

                    can_data[k] = raw_data[i][j][k][0:7].numpy()
                    dr_data[k] = raw_data[i][j][k][7:9].numpy()
                    can_yaw_rate[k] = raw_data[i][j][k][4].numpy()
                    can_wheel_speed[k] = raw_data[i][j][k][3].numpy()
                    can_steering_ang[k] = raw_data[i][j][k][1].numpy()
                    can_steering_spd[k] = raw_data[i][j][k][2].numpy()
                    can_lateral_accel[k] = raw_data[i][j][k][6].numpy()
                    can_longitudinal_accel[k] = raw_data[i][j][k][5].numpy()
                    dr_x[k] = raw_data[i][j][k][7].numpy()
                    dr_y[k] = raw_data[i][j][k][8].numpy()
                
            can_data_list[i] = can_data
            dr_data_list[i] = dr_data
            can_steering_ang_list[i] = can_steering_ang
            can_steering_spd_list[i] = can_steering_spd
            can_wheel_speed_list[i] = can_wheel_speed
            can_yaw_rate_list[i] = can_yaw_rate
            can_longitudinal_accel_list[i] = can_longitudinal_accel
            can_lateral_accel_list[i] = can_lateral_accel
            dr_x_list[i] = dr_x
            dr_y_list[i] = dr_y
            
        input_can_data = torch.tensor(can_data_list[:,0:50,1:], dtype=torch.float)
        input_dr_data = torch.tensor(dr_data_list[:,50:110,:], dtype=torch.float)

        input_can_steering_ang = torch.tensor(can_steering_ang_list[:,0:50,:], dtype=torch.float)
        input_can_steering_spd = torch.tensor(can_steering_spd_list[:,0:50,:], dtype=torch.float)
        input_can_wheel_speed = torch.tensor(can_wheel_speed_list[:,0:50,:], dtype=torch.float)
        input_can_yaw_rate = torch.tensor(can_yaw_rate_list[:,0:50,:], dtype=torch.float)
        input_can_longitudinal_accel = torch.tensor(can_longitudinal_accel_list[:,0:50,:], dtype=torch.float)
        input_can_lateral_accel = torch.tensor(can_lateral_accel_list[:,0:50,:], dtype=torch.float)
        
        input_dr_x = torch.tensor(dr_x_list[:,50:110,:], dtype=torch.float)
        input_dr_y = torch.tensor(dr_y_list[:,50:110,:], dtype=torch.float)
        # y = torch.stack(dr_data_list, dim=-1)
        # print(y.shape)
        
        # padding_mask = (y == 0).all(dim=-1)
        # padding_mask[0, -1] = False
        
        origin = torch.tensor([0, 0], dtype=torch.float)
        theta = torch.tensor([0], dtype=torch.float)
        scenario_id = torch.tensor([0], dtype=torch.int) 
        agent_id = torch.tensor([0], dtype=torch.int) 
        city = torch.tensor([0], dtype=torch.int) 
        
        return {
            "dr_x": input_dr_x,
            "dr_y": input_dr_y,
            "can_yaw_rate": input_can_yaw_rate,
            "can_wheel_speed": input_can_wheel_speed,
            "can_steering_spd": input_can_steering_spd,
            "can_steering_ang": input_can_steering_ang,
            "can_lateral_accel": input_can_lateral_accel,
            "can_longitudinal_accel": input_can_longitudinal_accel,
            'can_data': input_can_data,
            'dr_data': input_dr_data,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
        }
        pass