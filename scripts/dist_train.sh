env MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 NCCL_DEBUG=DEBUG \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
LSP_train.py \
    --logging_level DEBUG \
    --local_rank 0 \
    --config models/small/config.json \
    --model_name_or_path ./models/small \
    --init_checkpoint ./models/small/pytorch_model.bin \
    --train_input_file ./data/train.128len.db \
    --eval_input_file ./data/dummy_data.tsv \
    --output_dir ./models/output_model \
    --num_optim_steps 1000 \
    --valid_step 100 \
    --warmup_steps 16