"""
**preprocess.py Module**

Script to pre-process data files into Transformer compatible batches and batch iterators

"""
import logging
import dill as pickle
from pathlib import Path
from utils import *

import torch
import spacy
import pandas as pd
# conda install -c pytorch torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Class to generate pre-processed dataset into batches and interators for training, validation, testing sets.
    Module uses torchtext to generate dataset and iterators.
    The default tokenizer is 'spacy'

    """
    def __init__(self, src_path, trg_path, tokenizer='spacy', src_lang='de', trg_lang='en'):
        """

        :param src_path: path to src data file
        :param trg_path: path to trg data file
        :param tokenizer: tokenizer to use
        :param src_lang: src language
        :param trg_lang: trg language
        """
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.src_tokenizer = get_tokenizer(tokenizer, language=src_lang)
        self.trg_tokenizer = get_tokenizer(tokenizer, language=trg_lang)
        self.raw_data = {src_lang: [line for line in lines_from_file_path(src_path)],
                         trg_lang: [line for line in lines_from_file_path(trg_path)]}
        self.src_field = None
        self.trg_field = None

    def partition_raw_data(self, num_sents):
        """
        Partition dataset to specified number o sentences to create toy dataset

        :param num_sents: specify number of sentences to use for toy dataset
        :return:
        """
        logger.info(f'Partition raw data to {num_sents} sentences.')
        self.raw_data[self.src_lang] = self.raw_data[self.src_lang][:num_sents]
        self.raw_data[self.trg_lang] = self.raw_data[self.trg_lang][:num_sents]

    def make_train_val_test_splits(self, max_len, max_diff, test_split_size=0.2):
        """
        Create training, validation, and test splits

        :param max_len: maximum number of tokens in the sentence
        :param max_diff: maximum factor of difference in number of tokens between src and trg sentences
        :param test_split_size: proportion to hold out for test set
        :return:
        """
        logger.info(f'making train, val, test splits')
        # prune raw data for max seq len
        df = pd.DataFrame(self.raw_data, columns=[self.src_lang, self.trg_lang])
        df['src_len'] = df[self.src_lang].str.count(' ')
        df['trg_len'] = df[self.trg_lang].str.count(' ')
        df = df.query(f'src_len < {max_len} & trg_len < {max_len}')
        df = df.query(f'trg_len < src_len * {max_diff} & trg_len * {max_diff} > src_len')

        # make train, val, test splits
        # make train, val 80/20, then split val into val and split (0.5 split) for end result: 80/10/10
        train, val = train_test_split(df, test_size=test_split_size, shuffle=False)
        test, val = train_test_split(val, test_size=0.5, shuffle=False)

        return train, val, test

    @staticmethod
    def to_json(data_frame, path, filename):
        """
        Save data farm to json

        :param data_frame: pandas dataframe
        :param path: path to data directory
        :param filename: filename for the .json file
        :return:
        """
        data_frame.to_json(path / filename, orient='records', lines=True)

    def make_datasets(self, data_path, train_file, val_file, test_file, max_len, pkl_path, src_pkl_file='de_Field.pkl',
                      trg_pkl_file='en_Field.pkl', sos='<sos>', eos='<eos>', saved_field=False):
        """
        Create training, validation, and test datasets from json files in TabularDataset format.

        :param data_path: path to data directory
        :param train_file: training json file (should be the same as the file saved in to_json function)
        :param val_file: validation json file (should be the same as the file saved in to_json function)
        :param test_file: test json file (should be the same as the file saved in to_json function)
        :param max_len: maximum number of tokens in the sentence
        :param pkl_path: path to pickle file directory
        :param src_pkl_file: src pickle filename
        :param trg_pkl_file: trg pickle filename
        :param sos: start of sentence token
        :param eos: end of sentence token
        :param saved_field: whether or not there's a saved torchtext.data.Field pickled file to be loaded
        :return:
        """

        if saved_field:
            logger.info(f'loading saved Fields from {src_pkl_file} and {trg_pkl_file}')
            with open(pkl_path / src_pkl_file, mode='rb') as src_pkl, \
                    open(pkl_path / trg_pkl_file, mode='rb') as trg_pkl:
                self.src_field = pickle.load(src_pkl)
                self.trg_field = pickle.load(trg_pkl)
            logger.info(f'len loaded src field vocab: {len(self.src_field.vocab)}')
        else:
            # Field already includes default pad_token='<pad>', unknown='<unk>'
            # batch_first=True so that dim 0 is batch size, Transformer expects inputs shape: (batch_size, seq len)
            self.src_field = Field(sequential=True, use_vocab=True, init_token=sos, eos_token=eos, fix_length=max_len,
                                   tokenize=self.src_tokenizer, lower=True, batch_first=True)
            self.trg_field = Field(sequential=True, use_vocab=True, init_token=sos, eos_token=eos, fix_length=max_len,
                                   tokenize=self.trg_tokenizer, lower=True, batch_first=True)

        # fields mapping
        fields = {self.src_lang: ('src', self.src_field), self.trg_lang: ('trg', self.trg_field)}

        # train, val, test datasets from the json files
        train_data, val_data, test_data = TabularDataset.splits(path=data_path, train=train_file, validation=val_file,
                                                                test=test_file, format='json', fields=fields)

        # build vocab from the train set unless loading from saved vocab
        # if no saved_field, pickle them for next time
        if not saved_field:
            self.src_field.build_vocab(train_data, min_freq=2)
            self.trg_field.build_vocab(train_data, min_freq=2)

            logger.info(f'saving Fields to {src_pkl_file} and {trg_pkl_file}')
            with open(pkl_path / src_pkl_file, mode='wb') as src_pkl, \
                    open(pkl_path / trg_pkl_file, mode='wb') as trg_pkl:
                pickle.dump(self.src_field, src_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.trg_field, trg_pkl, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f'checking that pickled field does the same thing as the original')
            with open(pkl_path / src_pkl_file, mode='rb') as src_pkl, \
                    open(pkl_path / trg_pkl_file, mode='rb') as trg_pkl:
                loaded_src_field = pickle.load(src_pkl)

            assert len(self.src_field.vocab) == len(loaded_src_field.vocab), 'pickled src vocab size not the same?'
            logger.info(f'original len src field vocab: {len(self.src_field.vocab)}')
            example_src = [['seit', 'damals', 'ist', 'er', 'auf', 'über', '10.000', 'punkte'],
                           ['seit', 'damals', 'ist', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                           ['er', 'auf', 'über', '10.000', 'punkte', 'oov', '.', '<pad>']]
            logger.debug(f'example_src {example_src}')

            # Test results of numericalization
            original_numericalization = self.src_field.numericalize(example_src)
            logger.debug(f'og {original_numericalization}')
            pickled_numericalization = loaded_src_field.numericalize(example_src)
            logger.debug(f'pickled {pickled_numericalization}')

            assert torch.all(torch.eq(original_numericalization, pickled_numericalization))

        return train_data, val_data, test_data

    @staticmethod
    def make_batch_iterators(train_data, val_data, test_data, batch_size, device):
        """
        Create batch iterators for training, validation, and testing datasets using torchtext.data.BucketIterator

        :param train_data: training set in TabularDataset object
        :param val_data: validation set in TabularDataset object
        :param test_data: test set in TabularDataset object
        :param batch_size: number of sentences in a batch
        :param device: cuda or cpu
        :return:
        """
        logger.info(f'making batch iterators from data sets')
        # batch iterator
        train_iterator, val_iterator, test_iterator = BucketIterator.splits((train_data, val_data, test_data),
                                                                            batch_size=batch_size, device=device,
                                                                            sort_key=lambda x: len(x.src), shuffle=True)

        return train_iterator, val_iterator, test_iterator
