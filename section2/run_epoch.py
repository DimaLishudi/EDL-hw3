from enum import Enum

import torch
import section2.dataset
from torch.utils.data import DataLoader
from section2.transformer import miniGPT2

from timeit import Timer


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


# Я сделал класс miniGPT2 в transformer.py вместо этой функции
def get_gpt2_model() -> torch.nn.Module:
    pass


def run_epoch(data_mode: DataMode, batch_size=128, **kwargs) -> None:
    if data_mode == DataMode.BRAIN:
        dataset = section2.dataset.BrainDataset(**kwargs)
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, pin_memory=True, num_workers=1, drop_last=True,
        )
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = section2.dataset.BigBrainDataset(**kwargs)
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, pin_memory=True, num_workers=1, drop_last=True,
            collate_fn=section2.dataset.collate_fn,
        )
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        dataset = section2.dataset.UltraDuperBigBrainDataset(**kwargs)
        sampler = section2.dataset.UltraDuperBigBrainSampler(batch_size, dataset.n_bins, len(dataset))
        dataloader = DataLoader(
            dataset, pin_memory=True, num_workers=1,
            collate_fn=section2.dataset.collate_fn,
            batch_sampler=sampler,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = miniGPT2(len(dataset.vocab)).to(device)
    epoch_size = len(dataset) // batch_size

    t = Timer(lambda: model(next(iter(dataloader)).to(device)))

    # warmup
    t.repeat(repeat=2, number=1)
    # calculations
    res = t.repeat(repeat=3*epoch_size, number=1)
    return res