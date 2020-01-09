#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
'''
 * @Desc: train GPT2 from scratch/ fine tuning.
          Modified based on Huggingface GPT-2 implementation
'''

import argparse
import datetime
import json
import logging
import os
import sys
import time
from os.path import join

import numpy as np
import torch
import tqdm
from torch.distributed import get_rank, get_world_size, barrier
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from data_loader import (BucketingDataLoader, DistributedBucketingDataLoader,
                         DynamicBatchingLoader)
from gpt2_training.distributed import (all_gather_list,
                                       all_reduce_and_rescale_tensors)
from gpt2_training.eval_utils import eval_model_loss
from gpt2_training.train_utils import (boolean_string,
                                       get_eval_list_same_length, load_model,
                                       set_lr)
from lsp_model import Adam, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

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
    logger.debug('>>> torch.distributed.init_process_group')
    torch.distributed.init_process_group(backend='nccl')
    logger.debug('<<< torch.distributed.init_process_group')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, bool(args.local_rank != -1), args.fp16))

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = join(args.output_dir,
                  'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                               args.train_batch_size, n_gpu,
                                               timestamp))
log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
if args.local_rank == -1 or get_rank() == 0:
    logger.debug('make output dir %s', output_dir)
    os.makedirs(output_dir, exist_ok=True)

args_dict = vars(args)
args_strings = []
for k, v in args_dict.items():
    args_strings.append('\t--{:28}  {}'.format(k, v))
args_strings = os.linesep + os.linesep.join(args_strings)
logger.info('Input Argument Information:%s', args_strings)


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
    logger.debug(
        'construct DistributedBucketingDataLoader for train ... '
        'rank=%s, world_size=%s, train_input_file=%s, train_batch_size=%s, train_batch_size=%s',
        get_rank(), get_world_size(), args.train_input_file, args.train_batch_size, args.max_seq_length
    )
    train_dataloader = DistributedBucketingDataLoader(
        get_rank(), get_world_size(),
        args.train_input_file, args.train_batch_size,
        args.max_seq_length)
    logger.debug('train_dataloader.keys: %s', train_dataloader._get_keys())

eval_dataloader_loss = DynamicBatchingLoader(
    args.eval_input_file, enc, args.normalize_data,
    args.eval_batch_size, args.max_seq_length)

eval_dataloader_gen = get_eval_list_same_length(
    args.eval_input_file, enc, args.eval_batch_size, True)


#########################################################################
# Prepare Model and Optimizer
##########################################################################
logger.debug('>>> load_model')
model = load_model(GPT2LMHeadModel(config), args.init_checkpoint,
                   args, verbose=True)
logger.debug('<<< load_model')
if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(
        params, float(torch.distributed.get_world_size()))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = %s', total_params)

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
else:
    optimizer = Adam(optimizer_grouped_parameters, args.learning_rate,
                     max_grad_norm=1.0)

# Parallel wrappers should only be applied to the model(s) AFTER the model(s) have been returned from amp.initialize.
if args.local_rank == -1:
    if n_gpu > 1:
        logging.info('data parallel because more than one gpu')
        model = DataParallel(model)

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


if args.local_rank != -1:
    logger.debug('barrier()')
    barrier()


while epoch < args.num_epochs:
    logger.info("epoch %d. global_step=%s", epoch, global_step)
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        # activate new training mode
        seq_len = batch[0].shape[1]
        batch = tuple(t.to(device) for t in batch)
        input_ids, position_ids, token_ids, label_ids, *_ = batch
        if args.no_token_id:
            token_ids = None
        loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
        ## 似乎这样才行，可能是 scalar
        if loss.shape:
            loss = loss.mean()
        if ppl.shape:
            ppl = ppl.mean()
        ## 而这样对于 proc 的 distribute 是有问题的！可能
        # if n_gpu > 1:
        #     loss = loss.mean()
        #     ppl = ppl.mean()
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

        if args.local_rank != -1:
            logger.debug("global_step %d: barrier", global_step)
            barrier()
        # FIXME: distributed 的时候，如果多 proc 一旦 使用 optimizer ，后续的 CUDA 操作就会 dead lock!!!!
        # gradient update
        logger.debug("global_step %d: gradient update. ", global_step)
        step += 1
        if step % args.gradient_accumulation_steps == 0:
            set_lr(optimizer, global_step,
                   args.lr_schedule, args.learning_rate,
                   args.warmup_steps, args.warmup_proportion,
                   config.n_embd, args.num_optim_steps)
            # TODO: 临时试试看，别忘了恢复上面的 comments

            # if args.local_rank != -1:
            #     grads = [p.grad.data for p in model.parameters()
            #              if p.requires_grad and p.grad is not None]
            #     all_reduce_and_rescale_tensors(grads, float(1))
            # TODO: 临时试试看，别忘了恢复上面的 comments

            logger.debug("global_step %d: optimizer.step()... ", global_step)
            optimizer.step()
            logger.debug("global_step %d: optimizer.zero_grad()... ", global_step)
            optimizer.zero_grad()
            logger.debug("global_step %d: global_step += 1", global_step)
            # TODO: 临时试试看，别忘了恢复上面的 comments
            global_step += 1

            # Print log info to file
            logger.debug("global_step %d: DUMMY DATA", global_step)
            x = torch.zeros(1)
            logger.debug("global_step %d: DUMMY DATA - x: %s", global_step, x)
            logger.debug("global_step %d: DUMMY DATA - to %s", global_step, device)
            x = x.to(device)
            logger.debug("global_step %d: DUMMY DATA - x: %s", global_step, x)
            logger.debug("global_step %d: 0001", global_step)
            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total
            # TODO: 临时试试看，别忘了恢复上面的 comments
            # n_token_real_all_proc = 1  # TODO: 假的，别忘记删除这一行
            # n_token_total_all_proc = 1  # TODO: 假的，别忘记删除这一行

            logger.debug("global_step %d: 0002", global_step)
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

            logger.debug("global_step %d: 0003", global_step)
            if global_step % args.valid_step == 0:
                if args.local_rank == -1 or get_rank() == 0:
                    # only rank 0 process evaluate
                    file_name = join(output_dir, f'GP2-pretrain-step-{global_step}.pkl')
                    torch.save(
                        {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                        for k, v in model.state_dict().items()},
                        file_name
                    )
                    logger.info("global_step %d: success saved to %s", global_step, file_name)

                    logger.info("global_step %d: >>> eval", global_step)
                    eval_loss, eval_ppl = eval_model_loss(
                        model, enc, eval_dataloader_loss, epoch, args)
                    logger.info("global_step %d: <<< eval", global_step)
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
                    model.train()

                logger.debug("global_step %d: 0004", global_step)

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
