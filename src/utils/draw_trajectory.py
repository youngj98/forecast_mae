import numpy as np
import cv2

class Drawer():
    def __init__(self):
        self.canvas = np.ones((1000, 2500, 3), dtype=np.uint8) * 255
        self.offset_x = 500
        self.offset_y = 500
        self.zoom_ratio = 10
    
    def add_line_text(self, text, line_start_point=(50, 50), line_end_point=(80, 50), line_color=(192, 192, 192), position=(90, 50)):
        # 캔버스에 라인, 텍스트 추가
        cv2.line(self.canvas, line_start_point, line_end_point, line_color, thickness=3)
        cv2.putText(self.canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
    def add_text(self, text, position=(50, 800)):
        # 캔버스에 텍스트 추가
        cv2.putText(self.canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    def draw_rect(self, top_left, bottom_right, color=(0, 0, 0)):
        cv2.rectangle(self.canvas, top_left, bottom_right, (0, 0, 0), 2)
        cv2.rectangle(self.canvas, top_left, bottom_right, color, -1)

    def draw_grid(self, line_color=(0, 0, 0), thickness=1, pxstep=100):
        for px in range(0, 2500, pxstep):
            cv2.line(self.canvas, (px, 0), (px, 1000), line_color, thickness)
        for px in range(0, 1000, pxstep):
            cv2.line(self.canvas, (0, px), (2500, px), line_color, thickness)
        
    def draw_trajectory(self, position_x_local, position_y_local, color=(192, 192, 192)):
        """ 기본 경로 그리기, 색상은 기본값으로 회색 """
        
        for x, y in zip(position_x_local, position_y_local):
            y = -y
            resize_x = int(x * self.zoom_ratio + self.offset_x)
            resize_y = int(y * self.zoom_ratio + self.offset_y)
            cv2.circle(self.canvas, (resize_x, resize_y), 5, color, -1)

            if x == position_x_local[0] and y == - position_y_local[0]:
                prev_x = x
                prev_y = y
                continue
            prev_x = int(prev_x * self.zoom_ratio + self.offset_x)
            prev_y = int(prev_y * self.zoom_ratio + self.offset_y)
            cv2.line(self.canvas, (prev_x, prev_y), (resize_x, resize_y), color, 3)
            prev_x = x
            prev_y = y
    
    def reduce_draw_trajectory(self, position_x_local, position_y_local, color=(192, 192, 192)):
        """ 기본 경로 그리기, 색상은 기본값으로 회색 """
        length = len(position_x_local)
        reduce_point = np.append(np.arange(0, length, 5), length - 1)       # 점들 0.5초 간격으로 샘플링해서 그리기
        reduce_x = position_x_local[reduce_point]
        reduce_y = position_y_local[reduce_point]
        
        for x, y in zip(reduce_x, reduce_y):
            y = - y
            resize_x = int(x * self.zoom_ratio + self.offset_x)
            resize_y = int(y * self.zoom_ratio + self.offset_y)
            cv2.circle(self.canvas, (resize_x, resize_y), 5, color, -1)

            if x == reduce_x[0] and y == - reduce_y[0]:
                prev_x = x
                prev_y = y
                continue
            prev_x = int(prev_x * self.zoom_ratio + self.offset_x)
            prev_y = int(prev_y * self.zoom_ratio + self.offset_y)
            cv2.line(self.canvas, (prev_x, prev_y), (resize_x, resize_y), color, 3)
            prev_x = x
            prev_y = y

    def save_plot(self, path):
        cv2.imwrite(path, self.canvas)

    def clear(self):
        self.canvas = np.ones((1000, 2500, 3), dtype=np.uint8) * 255