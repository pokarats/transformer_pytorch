import pickle
import time
import math
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from tqdm import tqdm
import argparse


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):
        # torchtext bucket iterator generates batches of shape (seq leng, N samples i.e. batch size)
        # Transformer expects (N batch size, seq len) shape inputs
        src = batch.src.transpose(0, 1).to(device)
        trg = batch.trg.transpose(0, 1).to(device)

        optimizer.zero_grad()

        # model to predict the <eos> token in trg but not have it be an input into our model
        # slice the <eos> token off the end of the sequence.
        # forward pass
        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim == trg vocab size]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]

        # Output is of shape (batch_size, trg len - 1, output_dim) but Cross Entropy Loss
        # doesn't take input in that form.
        # Need to reshape to: output seq * batch_size to send to cost function
        # also don't need <sos> token, so slice from pos 1:
        output = output.reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # torchtext bucket iterator generates batches of shape (seq leng, N samples i.e. batch size)
            # Transformer expects (N batch size, seq len) shape inputs
            src = batch.src.transpose(0, 1).to(device)
            trg = batch.trg.transpose(0, 1).to(device)

            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            # Output is of shape (batch_size, trg len - 1, output_dim) but Cross Entropy Loss
            # doesn't take input in that form.
            # Need to reshape to: output seq * batch_size to send to cost function
            # also don't need <sos> token, so slice from pos 1:
            output = output.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

