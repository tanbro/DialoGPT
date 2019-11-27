#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
'''
 * @Desc: train GPT2 from scratch/ fine tuning.
          Modified based on Huggingface GPT-2 implementation
'''

import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch

import numpy as np

from os.path import join
from torch.distributed import get_rank, get_world_size

from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Adam
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from gpt2_training.eval_utils import eval_model_loss

from data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader


from gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list


INF = 100000000
CACHE_EMPTY_STEP = 10000
EVAL_STEP = 100000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--num_epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('-l', '--logging_level', type=str, default='info', choices=[s.lower() for s in logging._nameToLevel.keys()],
                    help=f'logging level')
parser.add_argument('--model_name_or_path', type=str,
                    help='pretrained model name or path to local checkpoint')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)

parser.add_argument("--skip_eval", action='store_true',
                    help='If true, skip evaluation.')
parser.add_argument("--init_checkpoint", type=str)
parser.add_argument("--train_input_file", type=str)
parser.add_argument("--eval_input_file", type=str)
parser.add_argument("--continue_from", type=int, default=0)

parser.add_argument("--train_batch_size", type=int, default=4,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=10000,
                    help="how many optim steps between validations")
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--normalize_data", type=boolean_string, default=True)
parser.add_argument("--fp16", type=boolean_string, default=True)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument("--no_token_id", type=boolean_string, default=True)

parser.add_argument("--output_dir", type=str)
parser.add_argument("--log_dir", type=str)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

# distributed
parser.add_argument('--local_rank', type=int, default=-1,
                    help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')


# do normal parsing
args = parser.parse_args()

# logging config
logging.basicConfig(
    format='%(asctime)s - %(levelname)s [%(process)05d] - %(name)s -   %(message)s',
    level=logging._nameToLevel[args.logging_level.upper()]
)
logger = logging.getLogger(__name__)


if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            # PHILLY ENV special cases
            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY',
                              os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY',
                              os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)


assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
    'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size
                         // args.gradient_accumulation_steps)
logger.info('train batch size = {}, '
            'new train batch size (after gradient accumulation) = {}'.format(
                args.train_batch_size*args.gradient_accumulation_steps,
                args.train_batch_size))


if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    logging.debug('Initializes the distributed backend ...')
    logging.debug('env MASTER_ADDR=%s', os.getenv('MASTER_ADDR'))
    logging.debug('env MASTER_PORT=%s', os.getenv('MASTER_PORT'))
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    logging.debug('n_gpu of distributed: %s', n_gpu)
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(args.output_dir,
                  'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                               args.train_batch_size, n_gpu,
                                               timestamp))
log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))


#########################################################################
# Prepare Data Set
##########################################################################
enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

config = GPT2Config.from_json_file(
    join(args.model_name_or_path, 'config.json'))

if args.local_rank == -1:
    train_dataloader = BucketingDataLoader(args.train_input_file,
                                           args.train_batch_size,
                                           args.max_seq_length)
else:
    train_dataloader = DistributedBucketingDataLoader(
        get_rank(), get_world_size(),
        args.train_input_file, args.train_batch_size,
        args.max_seq_length)

eval_dataloader_loss = DynamicBatchingLoader(
    args.eval_input_file, enc, args.normalize_data,
    args.eval_batch_size, args.max_seq_length)

eval_dataloader_gen = get_eval_list_same_length(
    args.eval_input_file, enc, args.eval_batch_size, True)


#########################################################################
# Prepare Model and Optimizer
##########################################################################
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,
                   args, verbose=True)
if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(
        params, float(torch.distributed.get_world_size()))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    try:
        from apex import amp
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          bias_correction=False)
    # Allow Amp to perform casts as required by the opt_level
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level='O1',
        loss_scale='dynamic' if args.loss_scale == 0 else args.loss_scale
    )
    # Parallel wrappers should only be applied to the model(s) AFTER the model(s) have been returned from amp.initialize.
    if n_gpu > 1:
        logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
else:
    optimizer = Adam(optimizer_grouped_parameters, args.learning_rate,
                     max_grad_norm=1.0)

#########################################################################
# Training !
##########################################################################

logger.info("Training !")

if args.local_rank == -1 or get_rank() == 0:
    train_logger = open(join(log_dir, 'train_log.txt'), 'a+', buffering=1)
    eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
    print('epoch,global_step,step,mean_loss,mean_ppl,n_token_real,'
          'n_token_total,epoch_time', file=train_logger)
    print('epoch,global_step,step,eval_loss,eval_ppl', file=eval_logger)

global_step = 0
step = 0
epoch = 0

if args.continue_from:
    global_step = args.continue_from
    step = global_step*2 - 1


if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None

while epoch < args.num_epochs:
    logger.info("epoch %d. global_step=%s", epoch, global_step)
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        logger.debug("global_step %d: iterate train_dataloader", global_step)
        # activate new training mode
        seq_len = batch[0].shape[1]
        logger.debug("global_step %d: seq_len=%s", global_step, seq_len)
        logger.debug('batch:\t%s', batch)
        # FIXME: distributed 的时候，这里有问题！！！！
        logger.debug("global_step %d: batch to device %s ...", global_step, device)
        batch = tuple(t.to(device) for t in batch)
        logger.debug("global_step %d: batch extract ...", global_step)
        input_ids, position_ids, token_ids, label_ids, *_ = batch
        logger.debug("global_step %d: input_ids: \t%s", global_step, input_ids)
        logger.debug("global_step %d: position_ids: \t%s", global_step, position_ids)
        logger.debug("global_step %d: token_ids: \t%s", global_step, token_ids)
        logger.debug("global_step %d: label_ids: \t%s", global_step, label_ids)
        if args.no_token_id:
            logger.debug("global_step %d: token_ids = None", global_step)
            token_ids = None
        logger.debug("global_step %d: forward...", global_step)
        loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
        logger.debug("global_step %d: forward OK.", global_step)
        logger.debug("global_step %d: loss=%s", global_step, loss)
        logger.debug("global_step %d: ppl=%s", global_step, ppl)

        if loss.shape:
            logger.debug("global_step %d: loss.mean()", global_step)
            loss = loss.mean()
        if ppl.shape:
            logger.debug("global_step %d: ppl.mean()", global_step)
            ppl = ppl.mean()
        # if n_gpu > 1:
        #     loss = loss.mean()
        #     ppl = ppl.mean()
        logger.debug("global_step %d: loss = loss / (args.train_batch_size / input_ids.shape[0])", global_step)
        loss = loss / (args.train_batch_size / input_ids.shape[0])
        if args.fp16:
            # Amp loss.backward() becomes:
            logger.debug("global_step %d: loss.backward() ... ", global_step)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            logger.debug("global_step %d: loss.backward() Ok. ", global_step)
        else:
            loss.backward()

        tr_loss += float(loss.item()) * (args.train_batch_size / input_ids.shape[0])
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps
        if ppl.item() < INF:
            tr_ppl += ppl.item()
        else:
            tr_ppl += mean_ppl
        mean_ppl = tr_ppl / nb_tr_steps

        n_token_total += input_ids.shape[0] * input_ids.shape[1]
        n_token_real += (input_ids != 0).sum().item()

        # gradient update
        logger.debug("global_step %d: gradient update. ", global_step)
        step += 1
        if step % args.gradient_accumulation_steps == 0:
            set_lr(optimizer, global_step,
                   args.lr_schedule, args.learning_rate,
                   args.warmup_steps, args.warmup_proportion,
                   config.n_embd, args.num_optim_steps)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            logger.debug("global_step %d: optimizer.step()... ", global_step)
            optimizer.step()
            logger.debug("global_step %d: optimizer.zero_grad()... ", global_step)
            optimizer.zero_grad()
            logger.debug("global_step %d: global_step += 1", global_step)
            global_step += 1

            # Print log info to file
            logger.debug("global_step %d: Print log info to file ... 1", global_step)
            if args.local_rank != -1:
                # FIXME: distributed 的时候，这里有问题！！！！
                # 下面两行是单 GPU 的复制上来的，原本的有问题！
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total
                # logger.debug("global_step %d: Print log info to file ... 1.1", global_step)
                # logger.debug("get_world_size():\t%s", get_world_size())
                # logger.debug("mean_loss:\t%s", mean_loss)
                # mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                # logger.debug("global_step %d: Print log info to file ... 1.2", global_step)
                # mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                # logger.debug("global_step %d: Print log info to file ... 1.3 n_token_real=%s", global_step, n_token_real)
                # n_token_real_all_proc = sum(all_gather_list(n_token_real))
                # logger.debug("global_step %d: Print log info to file ... 1.4", global_step)
                # n_token_total_all_proc = sum(all_gather_list(n_token_total))
                # logger.debug("global_step %d: Print log info to file ... 1.5", global_step)
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total

            logger.debug("global_step %d: Print log info to file ... 2", global_step)

            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"tok/s: {n_token_real_all_proc/epoch_time/1024:.2f}k "
                        f"ppl: {mean_ppl:.2f} "
                        f"epoch: {epoch}"
                    )
                    pbar.update(1)
                print('{},{},{},{},{},{},{},{}'.format(
                    epoch+1, global_step+1, step+1, mean_loss, mean_ppl,
                    n_token_real_all_proc, n_token_total_all_proc, epoch_time),
                    file=train_logger)

            logger.debug("global_step %d: Print log info to file ... 3", global_step)

            if global_step % args.valid_step == 0:
                if args.local_rank == -1 or get_rank() == 0:
                    # only rank 0 process evaluate
                    logger.debug("global_step %d: torch.save ...", global_step)
                    torch.save(
                        {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                         for k, v in model.state_dict().items()},
                        join(output_dir,
                             f'GP2-pretrain-step-{global_step}.pkl'))

                    logger.debug("global_step %d: eval_model_loss ...", global_step)
                    eval_loss, eval_ppl = eval_model_loss(
                        model, enc, eval_dataloader_loss, epoch, args)
                    # enable generation step evaluation for now
                    # gen_response = eval_model_generation(
                    #     model, enc, eval_dataloader_gen, epoch, args)
                    '''
                    # probably use beam search only for test set
                    if False:
                        gen_response_beam = eval_model_generation(
                            model, enc, eval_dataloader_gen, epoch, args,
                            use_beam_search=True, beam_width=3)
                    '''
                    print('{},{},{},{},{}'.format(
                        epoch+1, global_step+1, step+1, eval_loss, eval_ppl),
                        file=eval_logger)
                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    logger.debug("global_step %d: back to train", global_step)
                    model.train()
            if global_step >= args.num_optim_steps:
                logger.debug("global_step %d: break! global_step >= args.num_optim_steps", global_step)
                break

        if (step+1) % CACHE_EMPTY_STEP == 0:
            logger.debug("global_step %d: torch.cuda.empty_cache()", global_step)
            torch.cuda.empty_cache()

        logger.debug("global_step %d: end of step!", global_step)
    # end for
    logger.debug("global_step %d: end of for loop", global_step)

    logger.debug("global_step %d: end of dataloader iteration!!!", global_step)

    if global_step >= args.num_optim_steps:
        break
    epoch += 1

logger.info("Training compelted!")


if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    train_logger.close()
    eval_logger.close()
