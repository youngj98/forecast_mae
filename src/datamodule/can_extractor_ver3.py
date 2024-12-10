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
        # can_data_list = np.empty((len(raw_data), 110, 7), dtype=np.float32)
        # dr_data_list = np.empty((len(raw_data), 110, 2), dtype=np.float32)
        # can_yaw_rate_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        # can_wheel_speed_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        # can_steering_spd_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        # can_steering_ang_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        # can_lateral_accel_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        # can_longitudinal_accel_list = np.empty((len(raw_data), 110, 1), dtype=np.float32)
        # dr_list = np.empty((len(raw_data), 110, 2), dtype=np.float32)
        # dr_y_list = np.empty((len(raw_data), 110, 2), dtype=np.float32)
        
        # print("raw_data", len(raw_data))
        # print("raw_data_shape", raw_data.shape)
        # print("raw_data_shape_i", raw_data[0].shape)
        # print("index", index)

        # data = raw_data[index] # 원래 값
        # data_agent = raw_data['agent'][0][0][0][index][0]
        data_ego_can = raw_data['ego'][0][0][0][index][0]
        # data_ego_can[0]: steer_angle_deg
        # data_ego_can[1]: wheel_speed_FL_mps
        # data_ego_can[2]: wheel_speed_FR_mps
        # data_ego_can[3]: wheel_speed_RL_mps
        # data_ego_can[4]: wheel_speed_RR_mps
        # data_ego_can[5]: yaw_rate_dps
        # data_ego_can[6]: longitudinal_accel_mps2
        # data_ego_can[7]: lateral_accel_mps2
        # data_ego_can[8]: steer_angle_speed_dps
        # data_ego_can[9]: accel_pedal_per
        # data_ego_can[10]: brake_prs_bar
        # data_ego_can[11]: turn_signal_L
        # data_ego_can[12]: turn_signal_R
        # data_ego_can[13]: turn_signal_Hazard
        # data_ego_can[14]: turn_signal_blink_L
        # data_ego_can[15]: turn_signal_blink_R
        data_ego_state = raw_data['ego'][0][0][1][index][0]
        # data_ego_state[0]: ego_state_position_m
        # data_ego_state[0][0]: ego_state_position_m x
        # data_ego_state[0][1]: ego_state_position_m y
        # data_ego_state[1]: ego_state_orientation_rad
        # data_ego_state[2]: ego_state_velocity_mps
        # data_ego_state[3]: ego_state_padding_mask
        
        data_ego_gnss = raw_data['ego'][0][0][2][index][0]
        # data_ego_gnss[0]: ego_gnss_latitude
        # data_ego_gnss[1]: ego_gnss_longitude
        # data_ego_gnss[2]: ego_gnss_heading
        
        # can_data = np.empty((110, 7), dtype=np.float32) # 원래 값
        can_data = np.empty((110, 6), dtype=np.float32) # 수정 값
        dr_data = np.empty((110, 2), dtype=np.float32)
        can_yaw_rate = np.empty((110, 1), dtype=np.float32)
        can_wheel_speed = np.empty((110, 1), dtype=np.float32)
        can_steering_spd = np.empty((110, 1), dtype=np.float32)
        can_steering_ang = np.empty((110, 1), dtype=np.float32)
        can_lateral_accel = np.empty((110, 1), dtype=np.float32)
        can_longitudinal_accel = np.empty((110, 1), dtype=np.float32)
        yaw = np.empty((110, 1), dtype=np.float32)
        # dr = np.empty((110, 2), dtype=np.float32)
        # dr_y = np.empty((110, 2), dtype=np.float32)
        # print(data)
        # for j in range(len(data)):
        #     for k in range(len(data[j])):
        #         data[j] = np.array(data[j], dtype=np.float32)
        #         data[j] = torch.tensor(data[j], dtype=torch.float)
                
        #         can_data[k] = data[j][k][0:7].numpy()
        #         dr_data[k] = data[j][k][7:9].numpy()
        #         can_yaw_rate[k] = data[j][k][4].numpy()
        #         can_wheel_speed[k] = data[j][k][3].numpy()
        #         can_steering_ang[k] = data[j][k][1].numpy()
        #         can_steering_spd[k] = data[j][k][2].numpy()
        #         can_lateral_accel[k] = data[j][k][6].numpy()
        #         can_longitudinal_accel[k] = data[j][k][5].numpy()
        #         yaw[k] = data[j][k][9].numpy()
        #         # dr[k] = raw_data[i][j][k][7].numpy()
        #         # dr_y[k] = raw_data[i][j][k][8].numpy()
        for i in range(len(data_ego_can)):
            data_ego_can[i] = np.array(data_ego_can[i], dtype=np.float32)
            data_ego_can[i] = torch.tensor(data_ego_can[i], dtype=torch.float)
        for i in range(len(data_ego_state)):
            data_ego_state[i] = np.array(data_ego_state[i], dtype=np.float32)
            data_ego_state[i] = torch.tensor(data_ego_state[i], dtype=torch.float)
        for i in range(len(data_ego_gnss)):
            data_ego_gnss[i] = np.array(data_ego_gnss[i], dtype=np.float32)
            data_ego_gnss[i] = torch.tensor(data_ego_gnss[i], dtype=torch.float)
            
        for k in range(len(data_ego_can[0][0])):
            can_data[k][0] = data_ego_can[0][0][k].numpy()
            can_data[k][1] = data_ego_can[8][0][k].numpy()
            can_data[k][2] = data_ego_can[1][0][k].numpy()
            can_data[k][3] = data_ego_can[5][0][k].numpy()
            can_data[k][4] = data_ego_can[6][0][k].numpy()
            can_data[k][5] = data_ego_can[7][0][k].numpy()
            dr_data[k][0] = data_ego_state[0][0][k].numpy()
            dr_data[k][1] = data_ego_state[0][1][k].numpy()
            can_yaw_rate[k] = data_ego_can[5][0][k].numpy() * np.pi / 180
            can_wheel_speed[k] = data_ego_can[1][0][k].numpy()
            can_steering_ang[k] = data_ego_can[0][0][k].numpy()
            can_steering_spd[k] = data_ego_can[8][0][k].numpy()
            can_lateral_accel[k] = data_ego_can[7][0][k].numpy()
            can_longitudinal_accel[k] = data_ego_can[6][0][k].numpy()
            yaw[k] = data_ego_state[1][0][k].numpy()
            # dr[k] = raw_data[i][j][k][7].numpy()
            # dr_y[k] = raw_data[i][j][k][8].numpy()
                
        # can_data_list = can_data
        # dr_data_list = dr_data
        # can_steering_ang_list = can_steering_ang
        # can_steering_spd_list = can_steering_spd
        # can_wheel_speed_list = can_wheel_speed
        # can_yaw_rate_list = can_yaw_rate
        # can_longitudinal_accel_list = can_longitudinal_accel
        # can_lateral_accel_list = can_lateral_accel
        # dr_x_list = dr_x
        # dr_y_list = dr_y
            
        input_can_data = torch.tensor(can_data[0:50,1:], dtype=torch.float)
        input_dr_data = torch.tensor(dr_data[50:110,:], dtype=torch.float)

        input_can_steering_ang = torch.tensor(can_steering_ang[0:50,:], dtype=torch.float)
        input_can_steering_spd = torch.tensor(can_steering_spd[0:50,:], dtype=torch.float)
        input_can_wheel_speed = torch.tensor(can_wheel_speed[0:50,:], dtype=torch.float)
        input_can_yaw_rate = torch.tensor(can_yaw_rate[0:50,:], dtype=torch.float)
        input_can_longitudinal_accel = torch.tensor(can_longitudinal_accel[0:50,:], dtype=torch.float)
        input_can_lateral_accel = torch.tensor(can_lateral_accel[0:50,:], dtype=torch.float)
        input_yaw = torch.tensor(yaw[0:50,:], dtype=torch.float)
        
        input_dr_x = torch.tensor(dr_data[0:50,:], dtype=torch.float)
        input_dr_y = torch.tensor(dr_data[50:110,:], dtype=torch.float)
        # print("input_dr_x", input_dr_x)
        # print("input_dr_x_shape", input_dr_x.shape)
        x_ctrs = input_dr_x[49,:]
        # y = torch.stack(input_dr_data, dim=-1)
        # print(y.shape)
        # padding_mask = np.empty((50, 1), dtype=np.bool)
        padding_mask = (input_dr_x == 0).all(dim=-1)
        # padding_mask = padding_mask.unsqueeze(0)
        # print("padding_mask", padding_mask)
        # print("padding_mask_shape", padding_mask.shape)
        padding_mask[-1] = False
        
        origin = torch.tensor([0, 0], dtype=torch.float)
        theta = torch.tensor([0], dtype=torch.float)
        scenario_id = torch.tensor([0], dtype=torch.int) 
        agent_id = torch.tensor([0], dtype=torch.int) 
        city = torch.tensor([0], dtype=torch.int)
        x_heading = torch.zeros((50, 1), dtype=torch.float)
        
        return {
            "x": input_dr_x,
            "y": input_dr_y,
            "x_centers": x_ctrs,
            "x_angles": x_heading,
            "can_yaw_rate": input_can_yaw_rate,
            "can_wheel_speed": input_can_wheel_speed,
            "can_steering_spd": input_can_steering_spd,
            "can_steering_ang": input_can_steering_ang,
            "can_lateral_accel": input_can_lateral_accel,
            "can_longitudinal_accel": input_can_longitudinal_accel,
            'can_data': input_can_data,
            'dr_data': input_dr_data,
            'yaw': input_yaw,
            "x_padding_mask": padding_mask,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
        }
        pass