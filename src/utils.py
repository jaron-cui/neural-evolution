import datetime
import os
import re

import torch
from pathlib import Path
from typing import Any, List, Tuple


class TrainingRecord:
    def __init__(self, training_record_directory: str):
        self.training_record_directory = Path(training_record_directory)
        self.dated_sub_folder = self.training_record_directory / str(datetime.date.today())
        self.run_sub_folder = self.dated_sub_folder / str(datetime.datetime.now().time().strftime("%H-%M-%S"))
        self.last_epoch = -1

    def save_checkpoint(self, checkpoint: List[Any], epoch: int):
        checkpoint_path = self.run_sub_folder / f'generation_{epoch}_survivors.pt'
        os.makedirs(self.run_sub_folder, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved a genome of a specimen from generation {epoch} at `{checkpoint_path}`.')

    @staticmethod
    def resume_from(run_sub_folder: str) -> Tuple['TrainingRecord', List[Any]]:
        path = os.path.normpath(run_sub_folder)
        components = path.split(os.sep)
        record = TrainingRecord(os.sep.join(components[:-2]))
        record.dated_sub_folder = record.training_record_directory / components[-2]
        record.run_sub_folder = Path(run_sub_folder)
        record.last_epoch = max([
            int(item.split('_')[1]) for item in os.listdir(run_sub_folder)
            if re.match(r'generation_[0-9]+_survivors.pt', item)
        ], default=-1)
        if record.last_epoch == -1:
            raise ValueError(f'Did not find a generation from which to resume in `{run_sub_folder}`.')
        return record, torch.load(
            record.run_sub_folder / f'generation_{record.last_epoch}_survivors.pt',
            weights_only=False
        )
