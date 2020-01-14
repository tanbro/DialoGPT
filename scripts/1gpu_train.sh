env NCCL_DEBUG=info \
python LSP_train.py \
    --logging_level info \
    --config models/small/config.json \
    --model_name_or_path models/small \
    --init_checkpoint models/small/pytorch_model.bin \
    --train_input_file data/train.128len.db \
    --eval_input_file data/dummy_data.tsv \
    --output_dir models/output_model \
    --num_optim_steps 1000 \
    --valid_step 100 \
    --warmup_steps 16 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1