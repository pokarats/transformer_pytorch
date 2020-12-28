import argparse
import logging
from pathlib import Path


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
    parser.add_argument('-epochs', type=int, help='number of epochs to train for', default=2)
    parser.add_argument('-d_model', type=int, help='d_model or hidden size', default=512)
    parser.add_argument('-d_ff', type=int, help='d_ff or hidden size of FFN sublayer', default=2048)
    parser.add_argument('-n_layers', type=int, help='number of encoder/decoder layers', default=6)
    parser.add_argument('-heads', type=int, help='number of attention heads', default=8)
    parser.add_argument('-dropout', type=float, help='value for dropout p parameter', default=0.1)
    parser.add_argument('-batch_size', type=int, help='number of samples per batch', default=128)
    # parser.add_argument('-print_every', type=int, help='number of epochs for interval printing', default=10)
    parser.add_argument('-lr', type=float, help='learning rate for gradient update', default=3e-4)
    parser.add_argument('-max_len', type=int, help='maximum number of tokens in a sentence', default=150)
    parser.add_argument('-num_sents', type=int, help='number of sentences to partition toy corpus', default=1024)
    parser.add_argument('-toy', type=bool, help='whether or not toy dataset', default=True)
    parser.add_argument('-debug', type=bool, help='turn logging to debug mode to display more info', default=False)
    parser.add_argument('-save_model', type=bool, help='True to save model checkpoint', default=False)
    parser.add_argument('-override', type=bool, help='override existing log file', default=True)

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
    debug = args.debug
    save_model = args.save_model
    override = args.override

    # hyper-parameters
    max_len = args.max_len
    batch_size = args.batch_size
    num_epochs = args.epochs
    d_model = args.d_model
    d_ff = args.d_ff
    nx_layers = args.n_layers
    num_heads = args.heads
    p_dropout = args.dropout
    learning_rate = args.lr
    clip = 1
    max_diff = 1.5

    # setup logging
    log_filename = str(log_path / 'train_model.log')
    model_log = logging.getLogger(__name__)
    logging_filemode = 'w' if override else 'a'
    logging.basicConfig(filename=log_filename, filemode=logging_filemode,
                        format='%(asctime)s %(name)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


if __name__ == '__main__':
    main()
