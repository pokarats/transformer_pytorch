import argparse
import logging
import random
from pathlib import Path
import math

import torch
import torch.optim as optim
import torch.nn as nn

import preprocess
import train_model
import transformer_model
from utils import *


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
    parser.add_argument('-d_model', type=int, help='d_model or hidden size', default=512)
    parser.add_argument('-d_ff', type=int, help='d_ff or hidden size of FFN sublayer', default=2048)
    parser.add_argument('-n_layers', type=int, help='number of encoder/decoder layers', default=6)
    parser.add_argument('-heads', type=int, help='number of attention heads', default=8)
    parser.add_argument('-dropout', type=float, help='value for dropout p parameter', default=0.1)
    parser.add_argument('-batch_size', type=int, help='number of samples per batch', default=128)
    parser.add_argument('-lr', type=float, help='learning rate for gradient update', default=3e-4)
    parser.add_argument('-max_len', type=int, help='maximum number of tokens in a sentence', default=150)
    parser.add_argument('-num_sents', type=int, help='number of sentences to partition toy corpus', default=1024)
    parser.add_argument('-toy', type=bool, help='whether or not toy dataset', default=True)
    parser.add_argument('-override', type=bool, help='override existing log file', default=True)
    parser.add_argument('-seed', type=int, help='seed for the iterator random shuffling repeat', default=1234)
    parser.add_argument('-sample_idx', type=int, help='index for a sample sentence pair example', default=8)

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
    toy = args.toy
    override = args.override
    seed = args.seed
    sample_idx = args.sample_idx

    # hyper-parameters
    max_len = args.max_len
    batch_size = args.batch_size
    d_model = args.d_model
    d_ff = args.d_ff
    nx_layers = args.n_layers
    num_heads = args.heads
    p_dropout = args.dropout
    learning_rate = args.lr
    max_diff = 1.5

    # setup logging
    log_filename = str(log_path / 'eval_model.log')
    eval_log = logging.getLogger(__name__)
    if override:
        logging_filemode = 'w+'
    else:
        logging_filemode = 'a+'
    logging.basicConfig(filename=log_filename, filemode=logging_filemode,
                        format='%(asctime)s %(name)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_log.info(f'device: {device}')

    # load saved Fields and rebuild dataset
    data = preprocess.Vocabulary(src_file, trg_file, tokenizer='spacy', src_lang=src_lang, trg_lang=trg_lang)
    if toy:
        data.partition_raw_data(num_sents)

    try:
        train_data, val_data, test_data = data.make_datasets(data_path, train_file='training.json', val_file='val.json',
                                                             test_file='test.json', max_len=max_len,
                                                             pkl_path=pkl_path, saved_field=True)
    except FileNotFoundError:
        eval_log.exception(f'.json or .pkl files not found! need to remake train, val, test split to .json')

        training, val, test = data.make_train_val_test_splits(max_len=max_len, max_diff=max_diff, test_split_size=0.2)

        data.to_json(training, data_path, 'training.json')
        data.to_json(val, data_path, 'val.json')
        data.to_json(test, data_path, 'test.json')

        train_data, val_data, test_data = data.make_datasets(data_path, train_file='training.json', val_file='val.json',
                                                             test_file='test.json', max_len=max_len,
                                                             pkl_path=pkl_path, saved_field=False)

    # sample sentences
    eval_log.info(f'testing a sample sentence pair from training_data')
    example_src = vars(train_data.examples[sample_idx])['src']
    example_trg = vars(train_data.examples[sample_idx])['trg']
    eval_log.info(f'example_src = {example_src}')
    eval_log.info(f'example_trg = {example_trg}')

    # re making batch iterators from the same random seed for shuffling as during training
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    train_iter, val_iter, test_iter = data.make_batch_iterators(train_data, val_data, test_data, batch_size=batch_size,
                                                                device=device)

    # data parameters for model set-ups
    src_vocab_size = len(data.src_field.vocab)
    trg_vocab_size = len(data.trg_field.vocab)
    src_pad_idx = data.src_field.vocab.stoi[data.src_field.pad_token]
    trg_pad_idx = data.trg_field.vocab.stoi[data.trg_field.pad_token]

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
    # CrossEntropyLoss has softmax built in, so no need for the final softmax layer in model architecture
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    # loading trained model checkpoint (model and optimizer state dicts)
    load_checkpoint(torch.load(model_path / 'transformer_model.pth.tar'), model, optimizer)

    # loss at testing time
    eval_log.info(f'evaluating model on test dataset')
    test_loss = train_model.evaluate(model, test_iter, criterion)
    eval_log.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    # checking a predicted sentence from saved model
    example_translation = translate_sentence(example_src, data.src_field, data.trg_field, model, device)
    print(f'predicted example_trg from trained model = {example_translation}')

    # calculating BLEU score on the test dataset
    blue_score = calc_bleu_score(test_data, model, data.src_field, data.trg_field, device)
    eval_log.info(f"Model BLEU score {blue_score * 100:.2f}")


if __name__ == '__main__':
    main()
