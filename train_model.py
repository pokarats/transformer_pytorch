"""
**train_model.py Module**

Wrapper module to train Transformer model and save model check points based on best validation loss.

This is the module to run to train the model. Model parameters can be configured/changed from default values
via command line options. Otherwise, default values are as per the "Attention is All You Need" paper:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
Attention Is All You Need. ArXiv:1706.03762 [Cs]. http://arxiv.org/abs/1706.03762

"""
import random

import dill as pickle
import time
import math
import transformer_model
from utils import *
import preprocess
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse


def train(model, iterator, optimizer, criterion, clip, device):
    """
    Function to train in batches

    :param model: model to be used for training, e.g. Transformer model
    :param iterator: torchtext.data.BucketIterator object
    :param optimizer: torch optimizer
    :param criterion: loss function
    :param clip: parameter for clip_grad_norm
    :param device: cpu or cuda
    :return: epoch loss
    """
    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(iterator):
        # Transformer expects (N batch size, seq len) shape inputs
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        optimizer.zero_grad()

        # model to predict the <eos> token in trg but not have it be an input into our model
        # slice the <eos> token off the end of the sequence.
        # forward pass
        output = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim == trg vocab size]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]

        # Output is of shape (batch_size, trg len - 1, output_dim) but Cross Entropy Loss
        # doesn't take input in that form.
        # Need to reshape to: output seq * batch_size to send to cost function
        # also don't need <sos> token, so slice trg from pos 1:
        output = output.reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += float(loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    run model in eval mode in batches

    :param model: model to be used for evaluation, e.g. Transformer model
    :param iterator: torchtext.data.BucketIterator object
    :param criterion: loss function
    :param device: cpu or cuda
    :return: epoch validation or test loss
    """
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # Transformer expects (N batch size, seq len) shape inputs
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output = model(src, trg[:, :-1])
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

            epoch_loss += float(loss.item())

    return epoch_loss / len(iterator)


def epoch_time(start_time: float, end_time: float):
    """
    Function to calculate elapsed time in mins and secs

    :param start_time: time function ended
    :param end_time: time function finished
    :return: elapsed minutes, elapsed seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, help='data directory', default='data')
    parser.add_argument('-log_path', type=str, help='log file directory', default='log')
    parser.add_argument('-pkl_path', type=str, help='pickle file directory', default='pkl')
    parser.add_argument('-model_path', type=str, help='saved models directory', default='saved_models')
    parser.add_argument('-src_data', type=str, help='src corpus filename', default='news-commentary-v8.de-en.de')
    parser.add_argument('-trg_data', type=str, help='trg corpus filename', default='news-commentary-v8.de-en.en')
    parser.add_argument('-src_lang', type=str, help='source language', default='de')
    parser.add_argument('-trg_lang', type=str, help='target language', default='en')
    parser.add_argument('-epochs', type=int, help='number of epochs to train for', default=25)
    parser.add_argument('-dmodel', type=int, help='d_model or hidden size', default=512)
    parser.add_argument('-dff', type=int, help='d_ff or hidden size of FFN sublayer', default=2048)
    parser.add_argument('-nlayers', type=int, help='number of encoder/decoder layers', default=6)
    parser.add_argument('-heads', type=int, help='number of attention heads', default=8)
    parser.add_argument('-dropout', type=float, help='value for dropout p parameter', default=0.1)
    parser.add_argument('-batch_size', type=int, help='number of samples per batch', default=48)
    # parser.add_argument('-print_every', type=int, help='number of epochs for interval printing', default=10)
    parser.add_argument('-lr', type=float, help='learning rate for gradient update', default=3e-4)
    parser.add_argument('-max_len', type=int, help='maximum number of tokens in a sentence', default=150)
    parser.add_argument('-num_sents', type=int, help='number of sentences to partition toy corpus', default=1024)
    parser.add_argument('-toy_run', type=bool, help='whether or not toy dataset', default=True)
    parser.add_argument('-debug', type=bool, help='turn logging to debug mode to display more info', default=False)
    parser.add_argument('-save_model', type=bool, help='True to save model checkpoint', default=True)
    parser.add_argument('-override', type=bool, help='override existing log file', default=True)
    parser.add_argument('-seed', type=int, help='seed for the iterator random shuffling repeat', default=1234)

    args = parser.parse_args()

    # file management
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / args.data_path
    log_path = project_dir / args.log_path
    pkl_path = project_dir / args.pkl_path
    model_path = project_dir / args.model_path

    # mkdir if dir donl't exist
    try:
        log_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'{log_path} is already there')
        pass
    else:
        print(f'{log_path} was created to store train_model.log file')

    try:
        pkl_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'{pkl_path} is already there')
        pass
    else:
        print(f'{pkl_path} was created to store .pkl files')

    try:
        model_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'{model_path} is already there')
        pass
    else:
        print(f'{model_path} was created to store checkpoint files')

    # parameters to run the script
    src_file = data_path / args.src_data
    trg_file = data_path / args.trg_data
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    num_sents = args.num_sents
    toy = args.toy_run
    debug = args.debug
    save_model = args.save_model
    override = args.override
    seed = args.seed

    # hyper-parameters
    max_len = args.max_len
    batch_size = args.batch_size
    num_epochs = args.epochs
    d_model = args.dmodel
    d_ff = args.dff
    nx_layers = args.nlayers
    num_heads = args.heads
    p_dropout = args.dropout
    learning_rate = args.lr
    clip = 1
    max_diff = 1.5

    # setup logging
    log_filename = str(log_path / 'train_model.log')
    model_log = logging.getLogger(__name__)
    if override:
        logging_filemode = 'w+'
    else:
        logging_filemode = 'a+'
    logging.basicConfig(filename=log_filename, filemode=logging_filemode,
                        format='%(asctime)s %(name)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    if debug:
        model_log.setLevel(logging.DEBUG)

    model_log.info(f'---------START----------')

    model_log.info(f'training parameters:\n'
                   f'max_len: {max_len}\t'
                   f'batch_size: {batch_size}\t'
                   f'num_epochs: {num_epochs}\n'
                   f'd_model: {d_model}\t'
                   f'd_ff: {d_ff}\t'
                   f'nx_layers: {nx_layers}\n'
                   f'num_heads: {num_heads}\t'
                   f'dropout: {p_dropout}\t'
                   f'learning rate: {learning_rate}\n'
                   f'clip: {clip}\t'
                   f'max_diff: {max_diff}\t'
                   f'toy run: {toy}\n'
                   f'override: {override}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_log.info(f'device: {device}')

    # load and pre-process dataset
    data = preprocess.Vocabulary(src_file, trg_file, tokenizer='spacy', src_lang=src_lang, trg_lang=trg_lang)
    if toy:
        data.partition_raw_data(num_sents)

    try:
        train_data, val_data, test_data = data.make_datasets(data_path, train_file='training.json', val_file='val.json',
                                                             test_file='test.json', max_len=max_len,
                                                             pkl_path=pkl_path, saved_field=True)
    except FileNotFoundError:
        model_log.exception(f'.json or .pkl files not found! need to remake train, val, test split to .json')

        training, val, test = data.make_train_val_test_splits(max_len=max_len, max_diff=max_diff, test_split_size=0.2)

        data.to_json(training, data_path, 'training.json')
        data.to_json(val, data_path, 'val.json')
        data.to_json(test, data_path, 'test.json')

        train_data, val_data, test_data = data.make_datasets(data_path, train_file='training.json', val_file='val.json',
                                                             test_file='test.json', max_len=max_len,
                                                             pkl_path=pkl_path)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    train_iter, val_iter, test_iter = data.make_batch_iterators(train_data, val_data, test_data,
                                                                batch_size=batch_size, device=device)

    # data parameters for model set-ups
    src_vocab_size = len(data.src_field.vocab)
    trg_vocab_size = len(data.trg_field.vocab)
    src_pad_idx = data.src_field.vocab.stoi[data.src_field.pad_token]
    trg_pad_idx = data.trg_field.vocab.stoi[data.trg_field.pad_token]

    # set up model
    best_valid_loss = float('inf')
    model = transformer_model.Transformer(src_vocab_size=src_vocab_size,
                                          trg_vocab_size=trg_vocab_size,
                                          src_pad_idx=src_pad_idx,
                                          trg_pad_idx=trg_pad_idx,
                                          d_model=d_model,
                                          nx_layers=nx_layers,
                                          n_heads=num_heads,
                                          d_ff=d_ff,
                                          dropout_p=p_dropout,
                                          max_length=max_len,
                                          device=device).to(device)

    # optimizer parameters are according to the paper
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    # instead of Noam lr scheduler/decay as in the paper, use PyTorch lr scheduler instead for simplicity
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    # CrossEntropyLoss has softmax built in, so no need for the final softmax layer in model architecture
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    # train model, track training and validation losses
    for epoch in tqdm(range(num_epochs)):

        start_time = time.time()
        model_log.info(f'start training model, epoch {epoch + 1}')
        train_loss = train(model, train_iter, optimizer, criterion, clip, device)
        model_log.info(f'start evaluating model on validation set, epoch {epoch + 1}')
        valid_loss = evaluate(model, val_iter, criterion, device)
        # update lr for next epoch with scheduler based on valid_loss stagnation
        scheduler.step(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if save_model:
            if valid_loss < best_valid_loss:
                model_log.info(f'saving best model checkpoint')
                best_valid_loss = valid_loss
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                save_checkpoint(checkpoint, model_path / 'transformer_model.pth.tar')

        model_log.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        model_log.info(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        model_log.info(f'\tVal. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    model_log.info(f'---------END----------')


if __name__ == '__main__':
    main()
