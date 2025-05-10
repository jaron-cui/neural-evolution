import datetime
import io
import os
import re

import logging
import torch
from pathlib import Path
from typing import Any, List, Tuple


class TrainingRecord:
    def __init__(
        self,
        training_record_directory: str,
        dated_sub_folder_name: str = None,
        run_sub_folder_name: str = None,
        last_epoch: int = -1
    ):
        if dated_sub_folder_name is None:
            dated_sub_folder_name = str(datetime.date.today())
        if run_sub_folder_name is None:
            run_sub_folder_name = str(datetime.datetime.now().time().strftime("%H-%M-%S"))

        self.training_record_directory = Path(training_record_directory)
        self.dated_sub_folder = self.training_record_directory / dated_sub_folder_name
        self.run_sub_folder = self.dated_sub_folder / run_sub_folder_name
        self.last_epoch = last_epoch

        file_handler = logging.FileHandler(self.run_sub_folder / 'log.txt')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    def save_checkpoint(self, checkpoint: List[Any], epoch: int):
        checkpoint_path = self.run_sub_folder / f'generation_{epoch}_survivors.pt'
        os.makedirs(self.run_sub_folder, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Saved a genome of a specimen from generation {epoch} at `{checkpoint_path}`.')

    @staticmethod
    def resume_from(run_sub_folder: str) -> Tuple['TrainingRecord', List[Any]]:
        path = os.path.normpath(run_sub_folder)
        components = path.split(os.sep)
        last_epoch = max([
            int(item.split('_')[1]) for item in os.listdir(run_sub_folder)
            if re.match(r'generation_[0-9]+_survivors.pt', item)
        ], default=-1)
        if last_epoch == -1:
            raise ValueError(f'Did not find a generation from which to resume in `{run_sub_folder}`.')

        record = TrainingRecord(os.sep.join(components[:-2]), components[-2], components[-1], last_epoch)

        return record, torch.load(
            record.run_sub_folder / f'generation_{record.last_epoch}_survivors.pt',
            weights_only=False
        )

#
# # Taken from https://github.com/tqdm/tqdm/issues/313#issuecomment-267959111
# class TqdmToLogger(io.StringIO):
#     """
#     Redirect tqdm output to a logger, simulating in-place updates.
#     """
#     def __init__(self, logger, level=logging.INFO):
#         super().__init__()
#         self.logger = logger
#         self.level = level
#         self._buffer = ''
#
#     def write(self, buf):
#         # Accumulate buffer
#         self._buffer += buf
#         # If there's a carriage return, treat it as an in-place update
#         if '\r' in buf or '\n' in buf:
#             self.flush()
#
#     def flush(self):
#         line = self._buffer.strip('\r\n')
#         if line:
#             self.logger.log(self.level, line)
#         self._buffer = ''
#
#
# tqdm_logging = TqdmToLogger(logging.getLogger())
logging.basicConfig(level=logging.INFO)

