import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl

class MetricLoggerCallback(pl.Callback):
    def __init__(self, save_dir="metrics", min_filename="min_values.txt"):
        super().__init__()
        self.save_dir = save_dir
        self.min_filename = min_filename
        os.makedirs(self.save_dir, exist_ok=True)
        self.min_filepath = os.path.join(self.save_dir, self.min_filename)
        
        self.train_losses = []
        self.val_losses = []
        self.metrics = {
            "minADE1": [],
            "minADE6": [],
            "minFDE1": [],
            "minFDE6": [],
            "MR": [],
        }
    
    def on_train_epoch_end(self, trainer, pl_module):
        # 에포크가 끝날 때마다 메트릭 저장
        for metric_name in self.metrics:
        # 'val_' 접두어를 붙여서 메트릭 값을 가져오기
            val_metric_value = trainer.callback_metrics.get(f'val_{metric_name}')
            
            # validation metric 기록
            if val_metric_value is not None:
                self.metrics[metric_name].append(val_metric_value.cpu().item())
            else:
                self.metrics[metric_name].append(np.nan)

    def on_train_end(self, trainer, pl_module):
        # 학습이 끝난 후 그래프 저장
        self.save_plots()
        
        # 학습이 끝난 후 최소값 텍스트 파일 저장
        self.save_min_values()

    def save_plots(self):
        # 손실에 대한 그래프 저장
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.legend()
        plt.title('Loss over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.save_dir, 'train_loss_graph.png'))
        plt.close()
        
        plt.figure()
        plt.plot(self.val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.save_dir, 'val_loss_graph.png'))
        plt.close()

    def save_min_values(self):
        # 각 메트릭 및 손실에 대해 최소값을 계산하여 텍스트 파일에 저장
        with open(self.min_filepath, 'w') as f:
            for metric_name, values in self.metrics.items():
                min_value = np.nanmin(values)
                f.write(f"min_{metric_name}: {min_value}\n")
            
            # 손실 최소값 저장
            min_train_loss = np.nanmin(self.train_losses)
            min_val_loss = np.nanmin(self.val_losses)
            f.write(f"min_train_loss: {min_train_loss}\n")
            f.write(f"min_val_loss: {min_val_loss}\n")

    def log_losses(self, train_loss, val_loss=None):
        # 에포크 종료 시 손실 값을 기록
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
        else:
            self.train_losses.append(np.nan)
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())
        else:
            self.val_losses.append(np.nan)
