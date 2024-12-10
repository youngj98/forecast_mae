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

# Drawer class for visualization
class Drawer():
    def __init__(self):
        self.canvas = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        self.offset_x = 200
        self.offset_y = 500
        self.zoom_ratio = 10

    def draw_trajectory(self, position_x_local, position_y_local, color=(192, 192, 192)):
        """ 기본 경로 그리기, 색상은 기본값으로 회색 """
        for x, y in zip(position_x_local, position_y_local):
            resize_x = int(x * self.zoom_ratio + self.offset_x)
            resize_y = int(y * self.zoom_ratio + self.offset_y)
            cv2.circle(self.canvas, (resize_x, resize_y), 3, color, -1)

            if x == position_x_local[0] and y == position_y_local[0]:
                prev_x = x
                prev_y = y
                continue
            prev_x = int(prev_x * self.zoom_ratio + self.offset_x)
            prev_y = int(prev_y * self.zoom_ratio + self.offset_y)
            cv2.line(self.canvas, (prev_x, prev_y), (resize_x, resize_y), color, 1)
            prev_x = x
            prev_y = y

    def save_plot(self, path):
        cv2.imwrite(path, self.canvas)

    def clear(self):
        self.canvas = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

def visualize_trajectories(trainer, predicted_trajectory, gt_trajectory, batch_idx):
    drawer = Drawer()
    epoch = trainer.current_epoch

    gt_position_x_local = gt_trajectory[0, :, 0].detach().cpu().numpy()
    gt_position_y_local = gt_trajectory[0, :, 1].detach().cpu().numpy()

    drawer.draw_trajectory(gt_position_x_local, gt_position_y_local, color=(255, 0, 0))
    
    pred_position_x_local = predicted_trajectory[0, :, 0].detach().cpu().numpy()
    pred_position_y_local = predicted_trajectory[0, :, 1].detach().cpu().numpy()
    drawer.draw_trajectory(pred_position_x_local, pred_position_y_local, color=(0, 0, 255))
    
    drawer.save_plot(f"./result/trajectory_{epoch}_{batch_idx}.png")
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
        
        # metrics = MetricCollection(
        #     {
        #         "Accuracy": Accuracy(task="binary"),
        #         # "Precision": Precision(task="binary"),
        #         # "Recall": Recall(task="binary"),
        #         # "F1": F1(),
        #         # "AUROC": AUROC(),
        #     }
        # )
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True           # 이따 출력 보고 확인
        )
        return predictions, prob

    def cal_loss(self, out, data, batch_idx):      # 이따 출력 보고 확인
        y_hat, pi = out["y_hat"], out["pi"]
        # y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y, y_others = data["y"][:, 0], data["y"][:, 1:]

        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        # others_reg_mask = ~data["x_padding_mask"][:, 1:, 50:]
        # others_reg_loss = F.smooth_l1_loss(
        #     y_hat_others[others_reg_mask], y_others[others_reg_mask]
        # )

        # loss = agent_reg_loss + agent_cls_loss + others_reg_loss
        loss = agent_reg_loss + agent_cls_loss

        if batch_idx % 100 == 0 and self.save_visualization:
            visualize_trajectories(self, y_hat_best, data["y"], batch_idx)
        
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            # "others_reg_loss": others_reg_loss.item(),
        }

    def training_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data, batch_idx)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data, batch_idx)
        metrics = self.val_metrics(out, data["y"][:, 0])
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

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        # self.submission_handler = SubmissionAv2(
        #     save_dir=save_dir, filename=f"forecast_mae_{timestamp}"
        # )

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])  # 이따 출력 보고 확인

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
