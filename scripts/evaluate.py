# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import preprocess_non_seq_data, preprocess_sequential_data
from model import FFN, RNN
from train_eval_func import det_ffnn, det_rnn, val_ffnn, val_rnn
from utils.codes import CODES
from utils.data import load_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("voiceActivityDetection")


def preprocess_data(
    model_type: str,
    data: Dict,
    subsample: float = 0.5,
    batch_size: int = 64,
    context: int = 2,
    seq_len: int = 8,
) -> Dict:
    if model_type == "FFN":
        loader_test, _ = preprocess_non_seq_data(
            data["testing"]["files"],
            data["training"]["labels"],
            subsample=subsample,
            context=context,
            batch_size=batch_size,
        )
        loader_val, _ = preprocess_non_seq_data(
            data["validation"]["files"],
            data["validation"]["labels"],
            subsample=subsample,
            context=context,
            batch_size=batch_size,
        )
    elif model_type == "RNN":
        loader_test, _ = preprocess_sequential_data(
            data["testing"]["files"],
            data["testing"]["labels"],
            subsample=0,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        loader_val, _ = preprocess_sequential_data(
            data["validation"]["files"],
            data["validation"]["labels"],
            subsample=0,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    else:
        logger.error("Model type is not FNN nor RNN")
    loaders = {
        "validation": loader_val,
        "testing": loader_test,
    }
    return loaders


def load_model(
    model_type: str,
    path_model: str,
    input_size: int,
    n_hidden: int = 150,
    n_output: int = 2,
):
    if model_type == "FFN":
        model = FFN(input_size, n_hidden, n_output)
        model.load_state_dict(torch.load(path_model))
    elif model_type == "RNN":
        model = RNN(input_size, n_hidden, n_output, 1)
        model.load_state_dict(torch.load(path_model))
    else:
        logger.error("Model type is not FNN nor RNN")

    return model


def evaluate_model(model_type: str, loaders: Dict, model, device: str):
    if model_type == "FFN":
        fpr_val, fnr_val, eer_val, acc_val = det_ffnn(
            loaders["validation"], model, device
        )
        fpr_test, fnr_test, eer_test, acc_test = det_ffnn(
            loaders["testing"], model, device
        )
        criterion = nn.CrossEntropyLoss().to(device)
        loss_test = val_ffnn(loaders["testing"], model, device, criterion)
    elif model_type == "RNN":
        fpr_val, fnr_val, eer_val, acc_val = det_rnn(
            loaders["validation"], model, device
        )
        fpr_test, fnr_test, eer_test, acc_test = det_rnn(
            loaders["testing"], model, device
        )
        criterion = nn.CrossEntropyLoss().to(device)
        loss_test = val_rnn(loaders["testing"], model, device, criterion)
    else:
        logger.error("Model type is not FNN nor RNN")

    print("Val: EER: {}, ACC: {}".format(eer_val, acc_val))
    print("Test: EER: {}, ACC: {}".format(eer_test, acc_test))

    fpr_val, fnr_val, fpr_test, fnr_test = (
        np.array(fpr_val),
        np.array(fnr_val),
        np.array(fpr_test),
        np.array(fnr_test),
    )

    plt.plot(fpr_val, fnr_val, label="Validation")
    plt.plot(fpr_test, fnr_test, label="Test")
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.title("DET curve of feed-forward neural network")
    plt.legend()

    print("Test: Loss: {}".format(loss_test.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_audio",
        type=str,
        help="Path to audio files.",
    )
    parser.add_argument(
        "--path_labels",
        type=str,
        help="Path to label files.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Either FFN or RNN.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
    )
    parser.add_argument(
        "--batch_size",
        type=float,
    )
    parser.add_argument(
        "--context",
        type=float,
    )
    parser.add_argument(
        "--seq_len",
        type=float,
    )
    parser.add_argument(
        "--path_model",
        type=str,
        help="Path to trained model.",
    )
    parser.add_argument(
        "--input_size",
        type=str,
        help="Input size for model definition.",
    )
    parser.add_argument(
        "--n_hidden",
        type=str,
        help="Hidden size for model definition.",
    )
    parser.add_argument(
        "--n_output",
        type=str,
        help="Output size for model definition.",
    )

    args = parser.parse_args()

    path_audio = Path(args.path_audio)
    path_labels = Path(args.path_labels)

    data = load_data(args.path_audio, args.path_labels, CODES)
    loaders = preprocess_data(
        args.model_type,
        data,
        args.subsample,
        args.batch_size,
        args.context,
        args.seq_len,
    )
    model = load_model(
        args.model_type, args.path_model, args.input_size, args.n_hidden, args.n_output
    )
