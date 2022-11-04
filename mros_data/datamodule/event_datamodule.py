from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from mros_data.datamodule.event_dataset import SleepEventDataset
from mros_data.utils import collate
from mros_data.utils import get_train_validation_test
from mros_data.utils.default_event_matching import get_overlapping_default_events


@dataclass
class SleepEventDataModule(LightningDataModule):
    # Partition specific
    data_dir: str
    n_test: int = 1000
    n_eval: int = 200
    seed: int = 1337

    # Dataset specific
    events: dict = None
    window_duration: int = None
    cache_data: bool = False
    default_event_window_duration: list = None
    event_buffer_duration: int = None
    factor_overlap: int = 2
    fs: int = None
    matching_overlap: float = None
    minimum_overlap: float = None
    n_jobs: int = None
    n_records: int = None
    picks: list = None
    transform: Callable = None
    scaling: str = None

    # Dataloader specific
    batch_size: int = 1
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).resolve()
        partitions = get_train_validation_test(
            self.data_dir, number_test=self.n_test, number_validation=self.n_eval, seed=self.seed
        )
        self.train_records = partitions["train"]
        self.eval_records = partitions["eval"]
        self.test_records = partitions["test"]
        self.n_classes = len(self.events.keys())
        self.n_channels = len(self.picks)
        self.window_size = self.window_duration * self.fs
        self.example_input_array = torch.randn(self.batch_size, self.n_channels, self.window_size)
        self.localizations_default = get_overlapping_default_events(
            window_size=self.window_size,
            default_event_sizes=[d * self.fs for d in self.default_event_window_duration],
            factor_overlap=self.factor_overlap,
        )

        self.dataset_kwargs = dict(
            events=self.events,
            window_duration=self.window_duration,
            cache_data=self.cache_data,
            default_event_window_duration=self.default_event_window_duration,
            event_buffer_duration=self.event_buffer_duration,
            fs=self.fs,
            localizations_default=self.localizations_default,
            matching_overlap=self.matching_overlap,
            minimum_overlap=self.minimum_overlap,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            picks=self.picks,
            transform=self.transform,
            scaling=self.scaling,
        )

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit":
            self.train = SleepEventDataset(self.train_records, subset="train", **self.dataset_kwargs)
            self.eval = SleepEventDataset(self.eval_records, subset="eval", **self.dataset_kwargs)
            self.output_dims = [self.batch_size] + self.train.output_dims
        elif stage == "test":
            self.test = SleepEventDataset(self.eval_records, subset="test", **self.dataset_kwargs)
            self.output_dims = [self.batch_size] + self.test.output_dims

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )


if __name__ == "__main__":

    dm = SleepEventDataModule("data/mros/processed")
    print(repr(dm))
    print(dm)
