import torch
import logging


logger = logging.getLogger(__name__)


def lines_from_file_path(path, strip=True):
    """

    :param strip:
    :param path: string or PosixPath to file
    :return: generator object where next is a single \n rstripped line from file
    """

    logger.info(f'processing lines from file: {path}')
    # need to specify newline explicitly for some files to be split on new lines properly
    with open(path, encoding='utf-8', mode='r', newline='\n') as file_handle:
        for line in file_handle:
            if strip:
                yield line.rstrip('\n')
            else:
                yield line


def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    logger.info(f'Saving checkpoint to {filename}')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    logger.info(f'Loading checkpoint')
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
