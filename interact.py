"""Script to interact with the DialoGPT models.

ref: https://github.com/andreamad8/DialoGPT2-Interact
"""

import argparse
import json
import logging
import os
import re
import socket
import subprocess as sp
import sys
from functools import partial
from importlib import import_module
from os.path import abspath, dirname, exists, join

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm, trange

from demo_utils import download_model_folder
from env import END_OF_TEXT_TOKEN
from gpt2_training.train_utils import (boolean_string,
                                       fix_state_dict_namespace,
                                       get_eval_list_same_length, load_model)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s [%(process)d](%(processName)s) - %(name)s -   %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def cut_seq_to_eos(sentence, eos_id, remove_id=[-1]):
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos_id:
            sent.append(s)
        else:
            break
    return sent


# FROM HUGGING FACE REPO
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits


def generate_next_token(model, input_ids, position_ids=None, token_type_ids=None, prev=None, temperature=1, top_k=0, top_p=0, past=None):
    with torch.no_grad():
        if not past:
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, past=past)
        else:
            hidden_states, past = model.transformer(prev, past=past)
        logits = model.lm_head(hidden_states)
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits.unsqueeze(0), dim=-1)
        prev = torch.multinomial(probs, num_samples=1)
        return prev, probs[0][prev], past


def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, temperature=1, top_k=0, top_p=0, length=20, past=None, device='cuda', eos_id=None):
    output = input_ids.new_zeros([input_ids.size(0), 0])
    prev = input_ids
    for i in range(length):
        prev, probs, past = generate_next_token(
            model, input_ids, position_ids,
            token_type_ids, prev, temperature, top_k, top_p, past
        )
        if eos_id is not None:
            tokens = prev[0].cpu()
            if tokens[0] == eos_id:
                break
        output = torch.cat((output, prev), dim=1)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, default='')
    parser.add_argument("--fp16", type=boolean_string, default=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)

    parser.add_argument("--generation_length", type=int, default=256)
    parser.add_argument("--max_history", type=int, default=4)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--tokenizer_class", type=str,
                        default='tokenizers.tokenization_cn_32k_v3_2:GPT2BPETokenizer_CN')
    parser.add_argument("--tokenizer_model", type=str,
                        default='/home/Public/data/gpt2/output/gpt2_huamei_corpus.bpe_src.small')

    args = parser.parse_args()
    return args


def get_tokenizer(args):
    tokenizer_class = args.tokenizer_class.strip()
    module_name, class_name = tokenizer_class.split(':')
    mod = import_module(module_name)
    clz = getattr(mod, class_name)
    return clz.from_pretrained(args.tokenizer_model)


def run_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load tokenizer
    logger.info('load tokenizer ...')
    tokenizer = get_tokenizer(args)  # GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    eos_id = tokenizer.encode(END_OF_TEXT_TOKEN)[-1]
    logger.info('EOS: %d', eos_id)

    # load the GPT-2 model
    logger.info('load the GPT-2 model ...')
    config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
    model = load_model(GPT2LMHeadModel(config), args.load_checkpoint, args, verbose=True)
    logger.info('load the GPT-2 model ok.')

    model.to(device)
    model.eval()

    terminating = False
    history = []
    while not terminating:
        raw_text = ''
        while not raw_text:
            try:
                raw_text = input(">>> ")
            except (KeyboardInterrupt, EOFError) as err:
                terminating = True
                print(type(err).__name__, file=sys.stderr)
                break
            else:
                raw_text = raw_text.strip()
                if not raw_text:
                    print('Prompt should not be empty!')
        if terminating:
            break

        with torch.no_grad():
            history.append(raw_text)
            context_tokens = sum([tokenizer.encode(h) + [eos_id] for h in history], [])  # + [eos_id]
            context_tokens = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = torch.arange(0, context_tokens.size(-1), dtype=torch.long, device=context_tokens.device)

            out = generate_sequence(
                model, context_tokens, position_ids=position_ids,
                length=args.generation_length, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                eos_id=eos_id
            )

            out = out.tolist()
            # text = tokenizer.decode(cut_seq_to_eos(out[0], eos_id))  # .encode('ascii', 'ignore').decode('ascii')
            text = tokenizer.decode(out[0])
            print(text)
            history.append(text)
            history = history[-(2*args.max_history+1):]


if __name__ == '__main__':

    # PYTHON_EXE = sys.executable
    # MODEL_FOLDER = './models'
    # DATA_FOLDER = './data'

    # if os.path.exists(MODEL_FOLDER):
    #     print('Found existing ./models folder, skip creating a new one!')
    #     os.makedirs(MODEL_FOLDER, exist_ok=True)
    # else:
    #     os.makedirs(MODEL_FOLDER)

    #########################################################################
    # Download Model
    #########################################################################
    # logger.info('Downloading models...')
    # download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)

    # model size:  could be one of 'small' (GPT2 with 117M), 'medium'(345M) or 'large' (1542M)
    # dataset: one of 'multiref' or 'dstc'
    # from_scratch: True : load model trained from scratch or False: load model trained from fine-tuning the GPT-2
    # target_folder = download_model(model_size='medium', dataset='multiref', from_scratch=False)
    # logger.info('Done!\n')
    exit(run_model(parse_args()))
