#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
#
# Please assign the DATA_FOLDER before running this scripts, the data, pre-trained model, fine-tuned model will be
# downloaded automatically to DATA_FOLDER

import os
import sys
import logging
import shlex
import shutil
from functools import partial

from demo_utils import download_model_folder
import argparse
import subprocess as sp


PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PYTHON_EXE = sys.executable
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

print(f'PROJECT_FOLDER = {PROJECT_FOLDER}')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='dummy',
                    help='choose from dummy, small and full')
dargs = parser.parse_args()

assert dargs.data == 'dummy' or dargs.data == 'small' or dargs.data == 'full' , \
    'The specified data option is not support!'


logging.basicConfig(
    format='%(asctime)s - %(levelname)s [%(process)d](%(processName)s) - %(name)s -   %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


if os.path.exists(MODEL_FOLDER):
    print(f'Found existing models folder at {MODEL_FOLDER}, skip creating a new one!')
    os.makedirs(MODEL_FOLDER, exist_ok=True)
else:
    os.makedirs(MODEL_FOLDER)

#########################################################################
# Download Model
#########################################################################
logger.info('Downloading models...')
download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)

# model size:  could be one of 'small' (GPT2 with 117M), 'medium'(345M) or 'large' (1542M)
# dataset: one of 'multiref' or 'dstc'
# from_scratch: True : load model trained from scratch or False: load model trained from fine-tuning the GPT-2
target_folder = download_model(model_size='small', dataset='multiref', from_scratch=False)
logger.info('Done!\n')


#########################################################################
# Prepare Data
#########################################################################
logger.info('Downloading and Extracting Data...')
if dargs.data == 'dummy':
    cmd = 'bash prepare4db.sh'
    print(cmd)
    sp.run(shlex.split(cmd), cwd=DATA_FOLDER, check=True)
elif dargs.data == 'small':
    cmd = 'make -j 8'
    print(cmd)
    sp.run(shlex.split(cmd), cwd='reddit_extractor', check=True)
    fname_src = os.path.join('reddit_extractor', 'data', 'out', 'train.tsv.gz')
    fname_dst = os.path.join(DATA_FOLDER, 'train.tsv.gz')
    shutil.copyfile(fname_src, fname_dst)
    cmd = 'gzip -d ./train.tsv.gz'
    print(cmd)
    sp.run(shlex.split(cmd), cwd=DATA_FOLDER, check=True)
elif dargs.data == 'full':
    cmd = 'make -j 8'
    print(cmd)
    sp.run(shlex.split(cmd), cwd='reddit_extractor', env={'SIZE':'full'}, check=True)
    fname = os.path.join('reddit_extractor', 'data', 'out', 'train.tsv.gz')
    cmd = f'gzip -d "{fname}"'
    print(cmd)
    sp.run(shlex.split(cmd), cwd=DATA_FOLDER, check=True)
else:
    raise ValueError('you need to implement your own data type, or use either dummy, small, or full')


logger.info('Preparing Data...')
data_path = os.path.join(DATA_FOLDER, 'train.tsv')
MAX_LEN = 128
data_db = f'{data_path[:-4]}.{MAX_LEN}len.db'
if os.path.isdir(data_db):
    print(f'{data_db} exists, skip prepro.py')
else:
    cmd = f'{PYTHON_EXE} prepro.py --corpus {data_path} --max_seq_len {MAX_LEN}'
    print(cmd)
    sp.run(shlex.split(cmd), cwd=PROJECT_FOLDER, check=True)
logger.info('Done!\n')

#########################################################################
# Train !
#########################################################################
logger.info('Generating training CMD!')
logger.info('If there is any problem, please copy (modify) and run command below')
logger.info('#########################################################################')
train_cmd = 'LSP_train.py'
args = [
    '--model_name_or_path', target_folder,
    '--init_checkpoint', os.path.join(target_folder, 'pytorch_model.bin'),
    '--train_input_file', data_db ,  # file from last step
    '--eval_input_file', './data/dummy_data.tsv',   # dummy test data
    '--output_dir', os.path.join(MODEL_FOLDER, 'output_model'),
    '--seed', '42',
    '--max_seq_length', '128',
    '--train_batch_size', '512',
    '--gradient_accumulation_steps', '8',
    '--eval_batch_size', '64',
    '--learning_rate', '1e-5',
    '--num_optim_steps', '10000',
    '--valid_step', '5000',
    '--warmup_steps', '4000',
    '--normalize_data', 'true',
    '--fp16', 'true',
    '--lr_schedule', 'noam',
    '--loss_scale', '0.0',
    '--no_token_id', 'true',
    '--pbar', 'true'
]

arg = ' '.join(args)
train_cmd = train_cmd + ' ' + arg
print(PYTHON_EXE + ' ' +train_cmd)
logger.info('#########################################################################')
with open('./output.log', 'wb') as f: 
    process = sp.Popen([PYTHON_EXE] + train_cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT, cwd=PROJECT_FOLDER)
    for line in iter(process.stdout.readline, b''): 
        sys.stdout.write(line.decode(sys.stdout.encoding)) 
        f.write(line)
logger.info('Done!\n')
