# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from data import preprocess_non_seq_data, preprocess_sequential_data
from model import FFN, RNN
from train_eval_func import (compute_eer, eer_acc_ffnn, eer_acc_rnn,
                             train_ffnn, train_rnn, val_ffnn, val_rnn)
from utils.codes import CODES
from utils.data import load_data

logger = logging.getLogger("voiceActivityDetection")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess_data(
    model_type: str,
    data: Dict,
    subsample: float = 0.5,
    batch_size: int = 64,
    context: int = 2,
    seq_len: int = 8,
) -> (Dict, int):
    if model_type == "FFN":
        loader_tr, input_size = preprocess_non_seq_data(
            data["training"]["files"],
            data["training"]["labels"],
            subsample=subsample,
            context=context,
            batch_size=batch_size,
        )
        loader_test, _ = preprocess_non_seq_data(
            data["testing"]["files"],
            data["testing"]["labels"],
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
        loader_tr, input_size = preprocess_sequential_data(
            data["training"]["files"],
            data["training"]["labels"],
            subsample=subsample,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        loader_test, _ = preprocess_sequential_data(
            data["testing"]["files"],
            data["testing"]["labels"],
            subsample=subsample,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        loader_val, _ = preprocess_sequential_data(
            data["validation"]["files"],
            data["validation"]["labels"],
            subsample=subsample,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    else:
        logger.error("Model type is not FNN nor RNN")
    loaders = {
        "training": loader_tr,
        "validation": loader_val,
        "testing": loader_test,
    }
    return loaders, input_size


def define_model(
    model_type: str,
    input_size: int,
    n_hidden: int = 150,
    n_output: int = 2,
    lr: float = 0.001,
    n_layers: int = 2,
):
    if model_type == "FFN":
        model = FFN(input_size, n_hidden, n_output).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif model_type == "RNN":
        model = RNN(input_size, n_hidden, n_output, n_layers).to(device)
        optimizer = torch.optim.SGD(RNN.parameters(), lr=lr)
    else:
        logger.error("Model type is not FNN nor RNN")

    criterion = nn.CrossEntropyLoss().to(device)

    return model, optimizer, criterion


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
        "--learning_rate",
        type=str,
        help="Learning rate for model definition.",
    )

    args = parser.parse_args()

    path_audio = Path(args.path_audio)
    path_labels = Path(args.path_labels)

    data = load_data(path_audio, path_labels, CODES)
    loaders, input_data = preprocess_data(
        args.model_type,
        data,
        args.subsample,
        args.batch_size,
        args.context,
        args.seq_len,
    )
    model, optimizer, criterion = define_model(
        args.model_type, args.input_size, args.n_hidden, args.learning_rate
    )
    # train
    epoch = 1
    losses, losses_val = [], []
    while True:
        if args.model_type == "FFN":
            loss = train_ffnn(loaders["training"], model, device, criterion, optimizer)
            losses.append(loss)
            print("Train: Epoch {}: loss: {}".format(epoch, loss.item()))
            loss_val = val_ffnn(loaders["validation"], model, device, criterion)
            losses_val.append(loss_val)
            print("Val: Epoch {}: loss: {}".format(epoch, loss_val.item()))
            loss_sorted = sorted(losses_val)
            eer_v, acc_v = eer_acc_ffnn(loaders["validation"], model, device)
            print("Val: Epoch {}: EER: {}, ACC: {}".format(epoch, eer_v, acc_v))
            eer_t, acc_t = eer_acc_ffnn(loaders["testing"], model, device)
            print("Test: Epoch {}: EER: {}, ACC: {}".format(epoch, eer_t, acc_t))

        elif args.model_type == "RNN":
            loss = train_rnn(loaders["training"], model, device, criterion, optimizer)
            losses.append(loss)
            print("Train: Epoch {}: loss: {}".format(epoch, loss.item()))
            loss_val = val_rnn(loaders["validation"], model, device, criterion)
            losses_val.append(loss_val)
            print("Val: Epoch {}: loss: {}".format(epoch, loss_val.item()))
            eer_v, acc_v = eer_acc_rnn(loaders["validation"], model, device)
            print("Val: Epoch {}: EER: {}, ACC: {}".format(epoch, eer_v, acc_v))
            eer_t, acc_t = eer_acc_rnn(loaders["testing"], model, device)
            print("Test: Epoch {}: EER: {}, ACC: {}".format(epoch, eer_t, acc_t))
            loss_sorted = sorted(losses_val)
        else:
            logger.error("Model type is not FNN nor RNN")

        if (len(losses_val) > 1) and (losses_val[-1] > losses_val[-2]):
            break
        else:
            torch.save(
                model.state_dict(), os.path.join(os.getcwd(), "model-" + str(epoch))
            )
            epoch += 1
