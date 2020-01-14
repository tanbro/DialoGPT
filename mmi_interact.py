"""MMI decoder for DialoGTP

ref: https://github.com/LHolten/DialoGTP-MMI-decoder/blob/master/interact.py
"""

import os

import torch
import torch.nn.functional as F

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# device for forward and backward model
device_f = 'cuda'
device_r = 'cpu'

# sampling parameters
num_samples = 10
top_k = 20
mmi_temperature = 0.5


model_name = 'small'
models_dir = 'models'


def main():

    model_dir_path = os.path.join(models_dir, model_name)

    config_path = os.path.join(model_dir_path, 'config.json')
    finetune_weights_path = os.path.join(model_dir_path, f'{model_name}_ft.pkl')
    tokenizer_vocab_path = os.path.join(model_dir_path, 'vocab.json')
    tokenizer_merges_path = os.path.join(model_dir_path, 'merges.txt')

    tokenizer = GPT2Tokenizer(tokenizer_vocab_path, tokenizer_merges_path)

    end_token = torch.tensor([[50256]], dtype=torch.long)

    weights = torch.load(finetune_weights_path)

    # fix misused key value
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)

    cfg = GPT2Config.from_json_file('medium/config.json')
    model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
    model.load_state_dict(weights)
    if device_f == 'cuda':
        model.half()
    model.to(device_f)

    with torch.no_grad():
        model.eval()


def _get_response(output_token, past):
    out = torch.tensor([[]], dtype=torch.long, device=device_f)

    while True:
        output_token, past = model.forward(output_token, past=past)
        output_token = output_token[:, -1, :].float()
        indices_to_remove = output_token < torch.topk(output_token, top_k)[0][..., -1, None]
        output_token[indices_to_remove] = -float('Inf')
        output_token = torch.multinomial(F.softmax(output_token, dim=-1), num_samples=1)

        out = torch.cat((out, output_token), dim=1)

        if output_token.item() == end_token.item():
            break

    return out, past


def _score_response(output_token, correct_token):
    inputs = torch.cat((output_token, correct_token), dim=1)
    mask = torch.full_like(output_token, -1, dtype=torch.long)
    labels = torch.cat((mask, correct_token), dim=1)

    loss, _, _ = reverse_model(inputs, labels=labels)

    return -loss.float()


if __name__ == "__main__":
    main()
