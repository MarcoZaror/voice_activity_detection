# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(path: Path, codes: List) -> np.ndarray:
    files = []
    for file in os.listdir(path):
        for code in codes:
            if file[:3] == code:
                with open(path / file, "rb") as f:
                    files.append(np.load(f))
    files_array = np.concatenate((files[0], files[1]))
    for i in range(2, len(files)):
        files_array = np.concatenate((files_array, files[i]))
    return files_array


def add_context(files: List, context: int = 2) -> np.ndarray:
    """
    Add c + x and c - x periods to each example
    """
    examples = np.array([])
    examples_with_context = []
    for i in range(context, len(files) - context):
        for c in range(-context, context + 1):
            examples = np.concatenate((examples, files[i + c]), axis=0)
        examples_with_context.append(examples)
        examples = np.array([])
    return examples_with_context


def preprocess_non_seq_data(
    files: list, labels: List, context: int, batch_size: int, subsample: float
) -> (DataLoader, int):
    """
    Take the data, process it and returns an iterator
    """
    files = add_context(files, context=context)
    files = np.array(files)
    labels = labels[context:-context]
    if subsample > 0:
        idx = np.random.permutation(len(files))
        subsample = int(subsample * len(files))
        files = files[idx]
        labels = labels[idx]
        files = files[:subsample]
        labels = labels[:subsample]
    files = torch.tensor(files, dtype=torch.float32)
    labels = torch.tensor(labels).squeeze().float()
    dataset = TensorDataset(files, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, files.shape[1]


def preprocess_sequential_data(
    files: List, labels: List, seq_len: int, batch_size: int, subsample: float
) -> (DataLoader, int):
    """
    Take the data and returns an iterator
    """
    if subsample > 0:
        files = files[: int(subsample * len(files))]
        labels = labels[: int(subsample * len(labels))]
    files_array = np.zeros((files.shape[0] - (seq_len - 1), seq_len, files.shape[1]))
    for i in range(files.shape[0] - (seq_len - 1)):
        files_array[i] = files[i : i + seq_len]
    labels_array = np.zeros((files.shape[0] - (seq_len - 1)))
    for i in range(labels.shape[0] - (seq_len - 1)):
        labels_array[i] = labels[i + (seq_len - 1)]
    files_array = torch.tensor(files_array, dtype=torch.float32)
    labels_array = torch.tensor(labels_array, dtype=torch.float32)
    dataset = TensorDataset(files_array, labels_array)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, files.shape[1]
