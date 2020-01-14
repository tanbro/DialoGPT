# interact with hmwebmix model

env NCCL_DEBUG=info \
python interact.py \
    --model_name_or_path output_model/345m-hmwebmix-bpe-v3.2/pre-train-345m-hmwebmix-bpe-v3.2 \
    --load_checkpoint output_model/345m-hmwebmix-bpe-v3.2/GPT2.0.0001.1.2gpu.2020-01-08161624/GP2-pretrain-step-1000000.pkl
