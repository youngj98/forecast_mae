import math
import numpy as np

class Rule_based():
    def __init__(self):
        self.init_x = 0.0
        self.init_y = 0.0
        self.dt = 0.1
        self.future_time = 60
        
    def cv_model(self, wheel_speed, init_yaw):
        yaw = init_yaw
        prev_x = self.init_x
        prev_y = self.init_y
        
        speed_x = wheel_speed * math.cos(yaw)
        speed_y = wheel_speed * math.sin(yaw)
        
        position = np.empty((self.future_time, 2), dtype=np.float32)
        
        for i in range(self.future_time):
            current_x = prev_x + speed_x * self.dt
            current_y = prev_y + speed_y * self.dt
            position[i][0] = current_x
            position[i][1] = current_y
            prev_x = current_x
            prev_y = current_y
        
        return position
    
    def ca_model(self, wheel_speed, longitudinal_accel, lateral_accel, init_yaw):
        yaw = init_yaw
        prev_x = self.init_x
        prev_y = self.init_y
        
        prev_speed_x = wheel_speed * math.cos(yaw)
        prev_speed_y = wheel_speed * math.sin(yaw)
        
        position = np.empty((self.future_time, 2), dtype=np.float32)
        
        for i in range(self.future_time):
            current_speed_x = prev_speed_x + longitudinal_accel * self.dt
            current_speed_y = prev_speed_y + lateral_accel * self.dt
            current_x = prev_x + (prev_speed_x + current_speed_x) / 2 * self.dt
            current_y = prev_y + (prev_speed_y + current_speed_y) / 2 * self.dt
            position[i][0] = current_x
            position[i][1] = current_y
            
            prev_x = current_x
            prev_y = current_y
            prev_speed_x = current_speed_x
            prev_speed_y = current_speed_y

        return position
    
    def ctrv_model(self, wheel_speed, yaw_rate, init_yaw):
        prev_yaw = init_yaw
        prev_x = self.init_x
        prev_y = self.init_y
        
        prev_speed_x = wheel_speed * math.cos(prev_yaw)
        prev_speed_y = wheel_speed * math.sin(prev_yaw)
        
        position = np.empty((self.future_time, 2), dtype=np.float32)
        
        for i in range(self.future_time):
            current_yaw = yaw_rate * self.dt
            current_speed_x = prev_speed_x * math.cos(current_yaw) - prev_speed_y * math.sin(current_yaw)
            current_speed_y = prev_speed_x * math.sin(current_yaw) + prev_speed_y * math.cos(current_yaw)
            current_x = prev_x + (prev_speed_x + current_speed_x) / 2 * self.dt
            current_y = prev_y + (prev_speed_y + current_speed_y) / 2 * self.dt
            position[i][0] = current_x
            position[i][1] = current_y
            
            prev_x = current_x
            prev_y = current_y
            prev_speed_x = current_speed_x
            prev_speed_y = current_speed_y

        return position
    
    def ctra_model(self, wheel_speed, longitudinal_accel, lateral_accel, yaw_rate, init_yaw):
        prev_yaw = init_yaw
        prev_x = self.init_x
        prev_y = self.init_y
        
        prev_speed_x = wheel_speed * math.cos(prev_yaw)
        prev_speed_y = wheel_speed * math.sin(prev_yaw)
        
        position = np.empty((self.future_time, 2), dtype=np.float32)
        
        for i in range(self.future_time):
            current_yaw = yaw_rate * self.dt
            current_speed_x = (prev_speed_x + longitudinal_accel * self.dt) * math.cos(current_yaw) - (prev_speed_y + lateral_accel * self.dt) * math.sin(current_yaw)
            current_speed_y = (prev_speed_x + longitudinal_accel * self.dt) * math.sin(current_yaw) + (prev_speed_y + lateral_accel * self.dt) * math.cos(current_yaw)
            current_x = prev_x + (prev_speed_x + current_speed_x) / 2 * self.dt
            current_y = prev_y + (prev_speed_y + current_speed_y) / 2 * self.dt
            position[i][0] = current_x
            position[i][1] = current_y
            
            prev_x = current_x
            prev_y = current_y
            prev_speed_x = current_speed_x
            prev_speed_y = current_speed_y

        return position
        