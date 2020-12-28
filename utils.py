import spacy
import torch
import logging
from torchtext.data import bleu_score

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


def translate_sentence(model, sentence, src_field, trg_field, device, src_lang='de', max_length=80):
    # Load german tokenizer
    spacy_de = spacy.load(src_lang)

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_de(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, src_field.init_token)
    tokens.append(src_field.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [src_field.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    # Tensor shape needs to be (batch size, seq len)
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

    outputs_token_indices = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs_token_indices).unsqueeze(0).to(device)

        with torch.no_grad():
            output, _ = model(sentence_tensor, trg_tensor)

        predicted_token_index = output.argmax(2)[:, -1].item()
        outputs_token_indices.append(predicted_token_index)

        if predicted_token_index == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    translated_sentence = [trg_field.vocab.itos[idx] for idx in outputs_token_indices]
    # remove start token
    return translated_sentence[1:]


def calc_bleu_score(data, model, src_field, trg_field, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)['src']
        trg = vars(example)['trg']

        prediction = translate_sentence(model, src, src_field, trg_field, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)
