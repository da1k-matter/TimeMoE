#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import pickle
import gzip
import yaml
import numpy as np

from .ts_dataset import TimeSeriesDataset


class GeneralDataset(TimeSeriesDataset):
    def __init__(self, data_path, streaming: bool = False):
        self.streaming = streaming
        self.num_tokens = None

        if streaming and data_path.endswith('.jsonl'):
            self.data_path = data_path
            self.offsets = []
            self.seq_lens = []
            cur_offset = 0
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.offsets.append(cur_offset)
                    obj = json.loads(line)
                    seq = obj.get('sequence', obj)
                    self.seq_lens.append(len(seq))
                    cur_offset = f.tell()
            self.data = None
        else:
            self.data = read_file_by_extension(data_path)

    def __len__(self):
        if self.streaming:
            return len(self.offsets)
        return len(self.data)

    def __getitem__(self, seq_idx):
        if self.streaming:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                f.seek(self.offsets[seq_idx])
                line = f.readline()
                obj = json.loads(line)
                seq = obj.get('sequence', obj)
                return seq
        else:
            seq = self.data[seq_idx]
            if isinstance(seq, dict):
                seq = seq['sequence']
            return seq

    def get_num_tokens(self):
        if self.num_tokens is None:
            if self.streaming:
                self.num_tokens = sum(self.seq_lens)
            else:
                self.num_tokens = sum([len(seq) for seq in self])
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        if self.streaming:
            return self.seq_lens[seq_idx]
        else:
            seq = self[seq_idx]
            return len(seq)

    @staticmethod
    def is_valid_path(data_path):
        if os.path.exists(data_path) and os.path.isfile(data_path):
            parts = data_path.split('.')
            if len(parts) == 0:
                return False
            suffix = parts[-1]
            if suffix in ('json', 'jsonl', 'npy', 'npy.gz', 'pkl'):
                return True
            else:
                return False
        else:
            return False


def read_file_by_extension(fn):
    if fn.endswith('.json'):
        with open(fn, encoding='utf-8') as file:
            data = json.load(file)
    elif fn.endswith('.jsonl'):
        data = read_jsonl_to_list(fn)
    elif fn.endswith('.yaml'):
        data = load_yaml_file(fn)
    elif fn.endswith('.npy'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npz'):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith('.npy.gz'):
        with gzip.GzipFile(fn, 'r') as file:
            data = np.load(file, allow_pickle=True)
    elif fn.endswith('.pkl') or fn.endswith('.pickle'):
        data = load_pkl_obj(fn)
    else:
        raise RuntimeError(f'Unknown file extension: {fn}')
    return data


def read_jsonl_to_list(jsonl_fn):
    with open(jsonl_fn, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]


def load_yaml_file(fn):
    if isinstance(fn, str):
        with open(fn, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config
    else:
        return fn


def load_pkl_obj(fn):
    out_list = []
    with open(fn, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                out_list.append(data)
            except EOFError:
                break
    if len(out_list) == 0:
        return None
    elif len(out_list) == 1:
        return out_list[0]
    else:
        return out_list
