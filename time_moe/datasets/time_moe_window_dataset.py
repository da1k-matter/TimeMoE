#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
import numpy as np

from time_moe.datasets.ts_dataset import TimeSeriesDataset


class TimeMoEWindowDataset:
    """
    A dataset class for generating non-overlapping sliding windows from a time series dataset.
    This is useful for training models that require fixed-length input sequences and corresponding labels.

    Attributes:
        dataset (TimeSeriesDataset): The underlying time series dataset.
        context_length (int): Length of the input context window.
        prediction_length (int): Length of the prediction window. Defaults to 0.
        window_size (int): Total size of the sliding window (context_length + prediction_length).
        window_size_plus_one (int): Total size of the sliding window plus one.
        stride (int): Step size for sliding the window. Defaults to window_size.
        sub_seq_indexes (list): List of tuples containing sequence indices and their corresponding offsets.

    Methods:
        __len__():
            Returns the total number of sliding windows in the dataset.
        __iter__():
            Iterates over the dataset, yielding one sliding window at a time.
        __getitem__(seq_idx):
            Retrieves a sliding window, its labels, and a loss mask.

    Example:
        >>> dataset = TimeSeriesDataset(...)  # Assume this is a predefined dataset
        >>> context_length = 10
        >>> prediction_length = 5
        >>> window_dataset = TimeMoEWindowDataset(dataset, context_length, prediction_length, target_cols=[3])
        >>> for sample in window_dataset:
        >>>     print(sample['input_ids'], sample['labels'], sample['loss_masks'])

    Args:
        target_cols: indices of columns to predict. Other columns will be treated
            as context only and ignored when computing loss.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        stride: int = None,
        input_size: int = 1,
        target_cols=None,
        **kwrags
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size
        self.input_size = input_size
        if target_cols is None:
            self.target_cols = list(range(input_size))
        else:
            if isinstance(target_cols, int):
                self.target_cols = [target_cols]
            else:
                self.target_cols = list(target_cols)

        num_seqs = len(self.dataset)
        iterator = range(num_seqs)
        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
        self.sub_seq_indexes = []
        for seq_idx in iterator:
            n_points = (
                self.dataset.get_sequence_length_by_idx(seq_idx) // self.input_size
            )
            # Skip sequences with fewer than 2 points
            if n_points < 2:
                continue
            self.sub_seq_indexes.append((seq_idx, 0))
            for offset_idx in range(
                self.stride, n_points - self.window_size_plus_one + 1, self.stride
            ):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        start = offset_i * self.input_size
        end = (offset_i + self.window_size_plus_one) * self.input_size
        seq = self.dataset[seq_i][start:end]
        seq = np.array(seq, dtype=np.float32).reshape(-1, self.input_size)

        loss_mask = np.zeros((seq.shape[0] - 1, self.input_size), dtype=np.int32)
        loss_mask[:, self.target_cols] = 1
        n_pad = self.window_size_plus_one - seq.shape[0]
        if n_pad > 0:
            seq = np.pad(seq, ((0, n_pad), (0, 0)), "constant", constant_values=0)
            loss_mask = np.pad(
                loss_mask, ((0, n_pad), (0, 0)), "constant", constant_values=0
            )

        return {"input_ids": seq[:-1], "labels": seq[1:], "loss_masks": loss_mask}


class UniversalTimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data with pack technique.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        context_length: int,
        prediction_length: int = 0,
        shuffle: bool = False,
        **kwrags
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size

        iterator = range(n_seqs)
        if shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)

        try:
            from IPython import get_ipython

            if get_ipython() is not None:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            iterator = tqdm(iterator, total=n_seqs)
        except ImportError:
            pass

        for seq_idx in iterator:
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )

                    # update states
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    # add the part of this seq to cur_window
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )

                    # update states
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)

                    # reset current window
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            # drop last batch for speed-up
            pass

    def __len__(self):
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][
                start_idx_in_seq : start_idx_in_seq + offset
            ]
            seq.append(part_seq)
        if len(seq) == 1:
            seq = seq[0]
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
        else:
            seq = np.concatenate(seq, axis=0, dtype=np.float32)
        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
        }
