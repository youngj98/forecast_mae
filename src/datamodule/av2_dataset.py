from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .can_extractor_ver2 import CANExtractor

import scipy.io as sio
import os


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cached_split: str = None,
        extractor: CANExtractor = None,
        data_file: str = None,
    ):
        super(Av2Dataset, self).__init__()

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
            self.extractor = extractor
            self.load = False
        else:
            raise ValueError("Either cached_split or extractor must be specified")

        # print(
        #     f"data root: {data_root}/{cached_split}, total number of files: {len(self.file_list)}"
        # )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        if self.load:
            data = torch.load(self.file_list[index])
        else:
            data = self.extractor.get_data(self.file_list[index])

        return data
    
    def load_data(self, data_root, data_file):
        data_root = Path(data_root)
        
        file_path = data_root / data_file
        mat_data = sio.loadmat(file_path)
        return mat_data


def collate_fn(batch):
    data = {}

    for key in [
        "can_yaw_rate",
        "can_wheel_speed",
        "can_steering_spd",
        "can_steering_ang",
        "can_lateral_accel",
        "can_longitudinal_accel",
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    if "x_scored" in batch[0]:
        data["x_scored"] = pad_sequence(
            [b["x_scored"] for b in batch], batch_first=True
        )

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)
        
    if batch[0]["padding_mask"] is not None:
        data["padding_mask"] = pad_sequence(
            [b["padding_mask"] for b in batch], batch_first=True, padding_value=True
        )

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    return data
