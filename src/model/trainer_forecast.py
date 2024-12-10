import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection, Accuracy, Precision, Recall

from src.metrics import MR, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2

from .model_forecast import ModelForecast

import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.utils.rule_based import Rule_based
from src.metrics.rule_based_metric import Rule_based_Metric
from src.utils.draw_trajectory import Drawer

save_dir = str("/home/ailab/git/forecast_yj/forecast_mae/result_batch_32_gpus_1_only_can_new/")
# Drawer class for visualization
# class Drawer():
#     def __init__(self):
#         self.canvas = np.ones((1000, 2500, 3), dtype=np.uint8) * 255
#         self.offset_x = 500
#         self.offset_y = 500
#         self.zoom_ratio = 10
    
#     def add_line_text(self, text, line_start_point=(50, 50), line_end_point=(80, 50), line_color=(192, 192, 192), position=(90, 50)):
#         # 캔버스에 라인, 텍스트 추가
#         cv2.line(self.canvas, line_start_point, line_end_point, line_color, thickness=3)
#         cv2.putText(self.canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
#     def add_text(self, text, position=(50, 800)):
#         # 캔버스에 텍스트 추가
#         cv2.putText(self.canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
#     def draw_rect(self, top_left, bottom_right, color=(0, 0, 0)):
#         cv2.rectangle(self.canvas, top_left, bottom_right, (0, 0, 0), 2)
#         cv2.rectangle(self.canvas, top_left, bottom_right, color, -1)

#     def draw_grid(self, line_color=(0, 0, 0), thickness=1, pxstep=100):
#         for px in range(0, 2500, pxstep):
#             cv2.line(self.canvas, (px, 0), (px, 1000), line_color, thickness)
#         for px in range(0, 1000, pxstep):
#             cv2.line(self.canvas, (0, px), (2500, px), line_color, thickness)
        
#     def draw_trajectory(self, position_x_local, position_y_local, color=(192, 192, 192)):
#         """ 기본 경로 그리기, 색상은 기본값으로 회색 """
        
#         for x, y in zip(position_x_local, position_y_local):
#             y = -y
#             resize_x = int(x * self.zoom_ratio + self.offset_x)
#             resize_y = int(y * self.zoom_ratio + self.offset_y)
#             cv2.circle(self.canvas, (resize_x, resize_y), 5, color, -1)

#             if x == position_x_local[0] and y == - position_y_local[0]:
#                 prev_x = x
#                 prev_y = y
#                 continue
#             prev_x = int(prev_x * self.zoom_ratio + self.offset_x)
#             prev_y = int(prev_y * self.zoom_ratio + self.offset_y)
#             cv2.line(self.canvas, (prev_x, prev_y), (resize_x, resize_y), color, 3)
#             prev_x = x
#             prev_y = y
    
#     def reduce_draw_trajectory(self, position_x_local, position_y_local, color=(192, 192, 192)):
#         """ 기본 경로 그리기, 색상은 기본값으로 회색 """
#         length = len(position_x_local)
#         reduce_point = np.append(np.arange(0, length, 5), length - 1)       # 점들 0.5초 간격으로 샘플링해서 그리기
#         reduce_x = position_x_local[reduce_point]
#         reduce_y = position_y_local[reduce_point]
        
#         for x, y in zip(reduce_x, reduce_y):
#             y = - y
#             resize_x = int(x * self.zoom_ratio + self.offset_x)
#             resize_y = int(y * self.zoom_ratio + self.offset_y)
#             cv2.circle(self.canvas, (resize_x, resize_y), 5, color, -1)

#             if x == reduce_x[0] and y == - reduce_y[0]:
#                 prev_x = x
#                 prev_y = y
#                 continue
#             prev_x = int(prev_x * self.zoom_ratio + self.offset_x)
#             prev_y = int(prev_y * self.zoom_ratio + self.offset_y)
#             cv2.line(self.canvas, (prev_x, prev_y), (resize_x, resize_y), color, 3)
#             prev_x = x
#             prev_y = y

#     def save_plot(self, path):
#         cv2.imwrite(path, self.canvas)

#     def clear(self):
#         self.canvas = np.ones((1000, 2500, 3), dtype=np.uint8) * 255

def visualize_trajectories(trainer, predicted_trajectory, past_trajectory, gt_trajectory, batch_idx, cv_position, ca_position, ctrv_position, ctra_position, vel, longi_acc, lat_acc, yaw_rate, yaw, val=0):
    drawer = Drawer()
    epoch = trainer.current_epoch

    pred_position_x_local = predicted_trajectory[0, :, 0].detach().cpu().numpy()
    pred_position_y_local = predicted_trajectory[0, :, 1].detach().cpu().numpy()
    # pred_position_x_local = predicted_trajectory[:, 0].detach().cpu().numpy()
    # pred_position_y_local = predicted_trajectory[:, 1].detach().cpu().numpy()

    gt_position_x_local = gt_trajectory[0, :, 0].detach().cpu().numpy()
    gt_position_y_local = gt_trajectory[0, :, 1].detach().cpu().numpy()
    # gt_position_x_local = gt_trajectory[:, 0].detach().cpu().numpy()
    # gt_position_y_local = gt_trajectory[:, 1].detach().cpu().numpy()
    
    past_position_x_local = past_trajectory[0, :, 0].detach().cpu().numpy()
    past_position_y_local = past_trajectory[0, :, 1].detach().cpu().numpy()
    
    drawer.draw_grid()
    drawer.draw_rect((30, 30), (380, 230), (255, 255, 255))
    drawer.draw_rect((30, 710), (460, 930), (255, 255, 255))
    drawer.draw_rect((450, 520), (550, 560), (255, 255, 255))

    drawer.draw_trajectory(past_position_x_local, past_position_y_local, color=(192, 192, 192))     # 회색, BGR
    drawer.draw_trajectory(cv_position[:, 0], cv_position[:, 1], color=(0, 255, 0))     # 초록색, BGR
    drawer.draw_trajectory(ca_position[:, 0], ca_position[:, 1], color=(0, 165, 255))     # 마젠타, BGR
    drawer.draw_trajectory(ctrv_position[:, 0], ctrv_position[:, 1], color=(47, 79, 79))     # 짙은 청록, BGR
    drawer.draw_trajectory(ctra_position[:, 0], ctra_position[:, 1], color=(128, 0, 128))     # 보라, BGR
    drawer.draw_trajectory(pred_position_x_local, pred_position_y_local, color=(0, 0, 255))     # 빨간색, BGR
    drawer.draw_trajectory(gt_position_x_local, gt_position_y_local, color=(255, 0, 0))     # 파란색, BGR
    
    drawer.add_line_text("GT", line_start_point=(50, 50), line_end_point=(90, 50), line_color=(255, 0, 0), position=(100, 60))
    drawer.add_line_text("Predicted Model", line_start_point=(50, 80), line_end_point=(90, 80), line_color=(0, 0, 255), position=(100, 90))
    drawer.add_line_text("CV Model", line_start_point=(50, 110), line_end_point=(90, 110), line_color=(0, 255, 0), position=(100, 120))
    drawer.add_line_text("CA Model", line_start_point=(50, 140), line_end_point=(90, 140), line_color=(0, 165, 255), position=(100, 150))
    drawer.add_line_text("CTRV Model", line_start_point=(50, 170), line_end_point=(90, 170), line_color=(47, 79, 79), position=(100, 180))
    drawer.add_line_text("CTRA Model", line_start_point=(50, 200), line_end_point=(90, 200), line_color=(128, 0, 128), position=(100, 210))
    
    drawer.add_text("(0, 0)", position=(465, 550))
    drawer.add_text("At (0, 0)", position=(50, 750))
    drawer.add_text("Velocity: {:.4f}".format(vel.item()), position=(50, 800))
    drawer.add_text("Longitudinal accel: {:.4f}".format(longi_acc.item()), position=(50, 830))
    drawer.add_text("Lateral accel: {:.4f}".format(lat_acc.item()), position=(50, 860))
    drawer.add_text("Yaw: {:.4f}".format(yaw.item()), position=(50, 890))
    drawer.add_text("Yaw rate: {:.4f}".format(yaw_rate.item()), position=(50, 920))
    
    if val == 1:
        # use save_dir
        drawer.save_plot(save_dir + f"val_trajectory_{epoch}_{batch_idx}.png")
        # drawer.save_plot(f"./eval_result_batch_8_gpus_1_csv/val_trajectory_{epoch}_{batch_idx}.png")
    else:
        drawer.save_plot(save_dir + f"trajectory_{epoch}_{batch_idx}.png")
        # drawer.save_plot(f"./eval_result_batch_8_gpus_1_csv/trajectory_{epoch}_{batch_idx}.png")
    drawer.clear()
    
def reduce_visualize_trajectories(trainer, predicted_trajectory, past_trajectory, gt_trajectory, batch_idx, cv_position, ca_position, ctrv_position, ctra_position, vel, longi_acc, lat_acc, yaw_rate, yaw, val=0):
    drawer = Drawer()
    epoch = trainer.current_epoch

    pred_position_x_local = predicted_trajectory[0, :, 0].detach().cpu().numpy()
    pred_position_y_local = predicted_trajectory[0, :, 1].detach().cpu().numpy()
    # pred_position_x_local = predicted_trajectory[:, 0].detach().cpu().numpy()
    # pred_position_y_local = predicted_trajectory[:, 1].detach().cpu().numpy()

    gt_position_x_local = gt_trajectory[0, :, 0].detach().cpu().numpy()
    gt_position_y_local = gt_trajectory[0, :, 1].detach().cpu().numpy()
    # gt_position_x_local = gt_trajectory[:, 0].detach().cpu().numpy()
    # gt_position_y_local = gt_trajectory[:, 1].detach().cpu().numpy()
    
    past_position_x_local = past_trajectory[0, :, 0].detach().cpu().numpy()
    past_position_y_local = past_trajectory[0, :, 1].detach().cpu().numpy()
    
    drawer.draw_grid()
    drawer.draw_rect((30, 30), (380, 230), (255, 255, 255))
    drawer.draw_rect((30, 710), (460, 930), (255, 255, 255))
    drawer.draw_rect((450, 520), (550, 560), (255, 255, 255))

    drawer.reduce_draw_trajectory(past_position_x_local, past_position_y_local, color=(192, 192, 192))     # 회색, BGR
    drawer.reduce_draw_trajectory(cv_position[:, 0], cv_position[:, 1], color=(0, 255, 0))     # 초록색, BGR
    drawer.reduce_draw_trajectory(ca_position[:, 0], ca_position[:, 1], color=(0, 165, 255))     # 마젠타, BGR
    drawer.reduce_draw_trajectory(ctrv_position[:, 0], ctrv_position[:, 1], color=(47, 79, 79))     # 짙은 청록, BGR
    drawer.reduce_draw_trajectory(ctra_position[:, 0], ctra_position[:, 1], color=(128, 0, 128))     # 보라, BGR
    drawer.reduce_draw_trajectory(pred_position_x_local, pred_position_y_local, color=(0, 0, 255))     # 빨간색, BGR
    drawer.reduce_draw_trajectory(gt_position_x_local, gt_position_y_local, color=(255, 0, 0))     # 파란색, BGR
    
    drawer.add_line_text("GT", line_start_point=(50, 50), line_end_point=(90, 50), line_color=(255, 0, 0), position=(100, 60))
    drawer.add_line_text("Predicted Model", line_start_point=(50, 80), line_end_point=(90, 80), line_color=(0, 0, 255), position=(100, 90))
    drawer.add_line_text("CV Model", line_start_point=(50, 110), line_end_point=(90, 110), line_color=(0, 255, 0), position=(100, 120))
    drawer.add_line_text("CA Model", line_start_point=(50, 140), line_end_point=(90, 140), line_color=(0, 165, 255), position=(100, 150))
    drawer.add_line_text("CTRV Model", line_start_point=(50, 170), line_end_point=(90, 170), line_color=(47, 79, 79), position=(100, 180))
    drawer.add_line_text("CTRA Model", line_start_point=(50, 200), line_end_point=(90, 200), line_color=(128, 0, 128), position=(100, 210))
    
    drawer.add_text("(0, 0)", position=(465, 550))
    drawer.add_text("At (0, 0)", position=(50, 750))
    drawer.add_text("Velocity: {:.4f}".format(vel.item()), position=(50, 800))
    drawer.add_text("Longitudinal accel: {:.4f}".format(longi_acc.item()), position=(50, 830))
    drawer.add_text("Lateral accel: {:.4f}".format(lat_acc.item()), position=(50, 860))
    drawer.add_text("Yaw: {:.4f}".format(yaw.item()), position=(50, 890))
    drawer.add_text("Yaw rate: {:.4f}".format(yaw_rate.item()), position=(50, 920))
    
    if val == 1:
        drawer.save_plot(save_dir + f"reduce_val_trajectory_{epoch}_{batch_idx}.png")
        # drawer.save_plot(f"./eval_result_batch_8_gpus_1_csv/reduce_val_trajectory_{epoch}_{batch_idx}.png")
    else:
        drawer.save_plot(save_dir + f"reduce_trajectory_{epoch}_{batch_idx}.png")
        # drawer.save_plot(f"./eval_result_batch_8_gpus_1_csv/reduce_trajectory_{epoch}_{batch_idx}.png")
    drawer.clear()
    

class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.net = ModelForecast(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
        )

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
            }
        )
        
        self.save_visualization = True
        
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob

    def cal_loss(self, out, data, batch_idx, val):
        rule_based_model = Rule_based()
        y_hat, pi = out["y_hat"], out["pi"]
        # y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        # y, y_others = data["y"][:, 0], data["y"][:, 1:]
        x = data["x"]
        y = data["y"]
        l2_norm = torch.norm(y_hat - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        # y_hat_best = y_hat[best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best, y)
        
        # l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        # best_mode = torch.argmin(l2_norm, dim=-1)
        # y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        # agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        # others_reg_mask = ~data["x_padding_mask"][:, 1:, 50:]
        # others_reg_loss = F.smooth_l1_loss(
        #     y_hat_others[others_reg_mask], y_others[others_reg_mask]
        # )

        # loss = agent_reg_loss + agent_cls_loss + others_reg_loss
        loss = agent_reg_loss + agent_cls_loss
        # loss = agent_reg_loss
        if batch_idx % 1 == 0 and self.save_visualization:
            cv_position = rule_based_model.cv_model(data['can_wheel_speed'][0][49], data['yaw'][0][49])
            ca_position = rule_based_model.ca_model(data['can_wheel_speed'][0][49], data['can_longitudinal_accel'][0][49], data["can_lateral_accel"][0][49], data['yaw'][0][49])
            ctrv_position = rule_based_model.ctrv_model(data['can_wheel_speed'][0][49], data["can_yaw_rate"][0][49], data['yaw'][0][49])
            ctra_position = rule_based_model.ctra_model(data['can_wheel_speed'][0][49], data['can_longitudinal_accel'][0][49], data["can_lateral_accel"][0][49], data["can_yaw_rate"][0][49], data['yaw'][0][49])
            
            Rule_based_Metric.save_metric(self, batch_idx, y, y_hat_best, cv_position, ca_position, ctrv_position, ctra_position, val, save_dir)
            
            visualize_trajectories(self, y_hat_best, x, y, batch_idx, cv_position, ca_position, ctrv_position, ctra_position, data['can_wheel_speed'][0][49], data['can_longitudinal_accel'][0][49], data["can_lateral_accel"][0][49], data["can_yaw_rate"][0][49], data['yaw'][0][49], val)
            reduce_visualize_trajectories(self, y_hat_best, x, y, batch_idx, cv_position, ca_position, ctrv_position, ctra_position, data['can_wheel_speed'][0][49], data['can_longitudinal_accel'][0][49], data["can_lateral_accel"][0][49], data["can_yaw_rate"][0][49], data['yaw'][0][49], val)
        
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            # "others_reg_loss": others_reg_loss.item(),
        }

    def training_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data, batch_idx, val=0)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.trainer.callbacks[3].log_losses(losses["loss"])

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data, batch_idx, val=1)
        # print("data_y: ", data["y"].shape)
        # print("out_y: ", out["y_hat"].shape)
        metrics = self.val_metrics(out, data["y"])
        # metrics = self.val_metrics(out["y_hat"][:, 0], data["y"][:, 0])

        self.log(
            "val/reg_loss",
            losses["reg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        
        # self.trainer.callbacks[3].log_losses(train_loss=None, val_loss=losses["loss"])

    def on_test_start(self) -> None:
        print("on_test_start")
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        # self.submission_handler = SubmissionAv2(
        #     save_dir=save_dir, filename=f"forecast_mae_{timestamp}"
        # )

    def test_step(self, data, batch_idx) -> None:
        rule_based_model = Rule_based()
        out = self(data)
        # self.submission_handler.format_data(data, out["y_hat"], out["pi"])
        
        y_hat, pi = out["y_hat"], out["pi"]
        x = data["x"]
        y = data["y"]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        
        cv_position = rule_based_model.cv_model(data['can_wheel_speed'][0][49], data['yaw'][0][49])
        ca_position = rule_based_model.ca_model(data['can_wheel_speed'][0][49], data['can_longitudinal_accel'][0][49], data["can_lateral_accel"][0][49], data['yaw'][0][49])
        ctrv_position = rule_based_model.ctrv_model(data['can_wheel_speed'][0][49], data["can_yaw_rate"][0][49], data['yaw'][0][49])
        ctra_position = rule_based_model.ctra_model(data['can_wheel_speed'][0][49], data['can_longitudinal_accel'][0][49], data["can_lateral_accel"][0][49], data["can_yaw_rate"][0][49], data['yaw'][0][49])
            
        Rule_based_Metric.save_metric(self, batch_idx, y, y_hat_best, cv_position, ca_position, ctrv_position, ctra_position, val=2, save_dir=save_dir)
            
        visualize_trajectories(self, y_hat_best, x, y, batch_idx, cv_position, ca_position, ctrv_position, ctra_position, data['can_wheel_speed'][0][49], data['can_longitudinal_accel'][0][49], data["can_lateral_accel"][0][49], data["can_yaw_rate"][0][49], data['yaw'][0][49], val=2)

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
