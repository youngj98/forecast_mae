from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .can_extractor_ver3 import CANExtractor

import scipy.io as sio
import os


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cached_split: str = None,
        extractor: CANExtractor = None,
        data_file: str = None,
        data_name: str = None,
    ):
        super(Av2Dataset, self).__init__()

        self.file_list = []
        self.data = self.load_data(data_root, data_file)
        self.data_name = data_name

        if cached_split is not None:
            self.data_folder = Path(data_root) / cached_split
            self.file_list = sorted(list(self.data_folder.glob("*.pt")))
            self.load = True
        elif extractor is not None:
            self.extractor = extractor
            self.data_folder = Path(data_root)
            print(f"Extracting data from {self.data_folder}")
            self.file_list = list(self.data_folder.rglob("*.parquet"))
            self.load = False
        elif data_file is not None:
            self.data_folder = Path(data_root)
            print(f"Extracting data from {self.data_folder}")
            self.extractor = CANExtractor()
            self.load = False
        else:
            raise ValueError("Either cached_split or extractor must be specified")

        # print(
        #     f"data root: {data_root}/{cached_split}, total number of files: {len(self.file_list)}"
        # )

    def __len__(self) -> int:
        # CanData = self.data["train_data"]
        # CanData = self.data[self.data_name] # 원래 값
        CanData = self.data['agent'][0][0][0][0][0][0] # 수정 값
        return CanData.size
        # return len(self.file_list)

    def __getitem__(self, index: int):
        if self.load:
            data = torch.load(self.file_list[index])
        else:
            # data = self.extractor.get_data(self.data['train_data'], index)
            # data = self.extractor.get_data(self.data[self.data_name], index)    # 원래 값
            data = self.extractor.get_data(self.data, index)    # 수정 값
            # print("data", data.keys())

        return data
    
    def load_data(self, data_root, data_file):
        data_root = Path(data_root)
        import os
        print(os.getcwd())
        file_path = data_root / data_file
        mat_data = sio.loadmat(file_path)
        return mat_data


def collate_fn(batch):
    data = {}

    for key in [
        "can_data",
        "dr_data",
        "x_centers",
        "x_angles",
        # "can_yaw_rate",
        # "can_wheel_speed",
        # "can_steering_spd",
        # "can_steering_ang",
        # "can_lateral_accel",
        # "can_longitudinal_accel",
        "x",
        "y",
        "x_padding_mask",
        "yaw",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)
    # data["can_data"] = torch.cat([b["can_data"] for b in batch], dim=0)
    # data["dr_data"] = torch.cat([b["dr_data"] for b in batch], dim=0)
    # data["can_yaw_rate"] = torch.cat([b["can_yaw_rate"] for b in batch], dim=0)
    # data["can_wheel_speed"] = torch.cat([b["can_wheel_speed"] for b in batch], dim=0)
    # data["can_steering_spd"] = torch.cat([b["can_steering_spd"] for b in batch], dim=0)
    # data["can_steering_ang"] = torch.cat([b["can_steering_ang"] for b in batch], dim=0)
    # data["can_lateral_accel"] = torch.cat([b["can_lateral_accel"] for b in batch], dim=0)
    # data["can_longitudinal_accel"] = torch.cat([b["can_longitudinal_accel"] for b in batch], dim=0)
    # data["dr_x"] = torch.cat([b["dr_x"] for b in batch], dim=0)
    # data["dr_y"] = torch.cat([b["dr_y"] for b in batch], dim=0)

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    return data
