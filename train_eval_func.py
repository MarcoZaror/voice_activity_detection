# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


def binarize_prediction(value: float) -> int:
    return 1 if value > 0.5 else 0


def compute_eer(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray
) -> (float, np.ndarray):
    """Returns equal error rate (EER) and the corresponding threshold."""
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]


def train_ffnn(
    loader: DataLoader, model, device: str, criterion: nn.CrossEntropy, optimizer
) -> float:
    """train feed forward neural network"""
    model.train()
    loss, batches = 0, 0
    for input, labels in loader:
        input = input.to(device)
        labels = labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        preds = model(input).squeeze()
        loss_batch = criterion(preds, labels)
        loss_batch.backward()
        optimizer.step()
        loss += loss_batch
        batches += 1
    return loss / batches


def val_ffnn(
    loader: DataLoader, model, device: str, criterion: nn.CrossEntropy
) -> float:
    """Compute validation loss for feedforward neural network"""
    model.eval()
    loss, batches = 0, 0
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            labels = labels.long()
            labels = labels.to(device)
            preds = model(input).squeeze()
            loss_batch_t = criterion(preds, labels)
            loss += loss_batch_t
            batches += 1
    return loss / batches


def eer_acc_ffnn(loader: DataLoader, model, device: str) -> (float, float):
    """Compute eer and accuracy for feedforward neural network"""
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            p = model(input).squeeze()
            preds.append(p[:, 1].flatten())
            labels_list.append(labels.long())
        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]
        fpr, tpr, thresholds = roc_curve(l_test, preds)
        eer, _ = compute_eer(fpr, tpr, thresholds)
        preds = [binarize_prediction(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return eer, acc


def det_ffnn(
    loader: DataLoader, model, device: str
) -> (np.ndarray, np.ndarray, float, float):
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            p = model(input).squeeze()
            preds.append(p[:, 1].flatten())
            labels_list.append(labels.long())
        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]
        fpr, tpr, thresholds = roc_curve(l_test, preds)
        fnr = 1 - tpr
        eer, _ = compute_eer(fpr, tpr, thresholds)
        preds = [binarize_prediction(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return fpr, fnr, eer, acc


def train_rnn(loader: DataLoader, model, device: str, criterion, optimizer) -> float:
    """Train recurrent neural network"""
    model.train()
    loss, batches = 0, 0
    for input, labels in loader:
        input = input.to(device)
        labels = labels.to(device)
        input = input.permute(1, 0, 2)
        batch_size = input.size()[1]
        optimizer.zero_grad()
        hidden = model.initHidden(batch_size)
        hidden = hidden.to(device)
        preds = model(input, hidden)
        loss_batch = criterion(preds, labels.long())
        loss_batch.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss += loss_batch
        batches += 1
    return loss / batches


def val_rnn(loader: DataLoader, model, device: str, criterion) -> float:
    """Compute validation loss for recurrent neural network"""
    model.eval()
    loss, batches = 0, 0
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            labels = labels.to(device)
            input = input.permute(1, 0, 2)
            batch_size = input.size()[1]
            hidden = model.initHidden(batch_size)
            hidden = hidden.to(device)
            preds = model(input, hidden)
            loss_batch = criterion(preds, labels.long())
            loss += loss_batch
            batches += 1
    return loss / batches


def eer_acc_rnn(loader: DataLoader, model, device: str) -> (float, float):
    """Compute eer and accuracy for feedforward neural network"""
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            input = input.permute(1, 0, 2)
            batch_size = input.size()[1]
            hidden = model.initHidden(batch_size)
            hidden = hidden.to(device)
            p = model(input, hidden)
            preds.append(p[:, 1].flatten())
            labels_list.append(labels.long())

        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]

        fpr, tpr, thresholds = roc_curve(l_test, preds)
        eer, _ = compute_eer(fpr, tpr, thresholds)
        preds = [binarize_prediction(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return eer, acc


def det_rnn(
    loader: DataLoader, model, device: str
) -> (np.ndarray, np.ndarray, float, float):
    preds = []
    labels_list = []
    with torch.no_grad():
        for input, labels in loader:
            input = input.to(device)
            input = input.permute(1, 0, 2)
            batch_size = input.size()[1]
            hidden = model.initHidden(batch_size)
            hidden = hidden.to(device)
            p = model(input, hidden)
            preds.append(p[:, 1].flatten())
            labels_list.append(labels.long())

        preds = [item.cpu().numpy() for sublist in preds for item in sublist]
        l_test = [int(item.cpu()) for sublist in labels_list for item in sublist]
        fpr, tpr, thresholds = roc_curve(l_test, preds)
        fnr = 1 - tpr
        eer, _ = compute_eer(fpr, tpr, thresholds)
        preds = [binarize_prediction(item) for item in preds]
        acc = accuracy_score(l_test, preds)
        return fpr, fnr, eer, acc
