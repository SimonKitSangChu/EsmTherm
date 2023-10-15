import argparse
from pathlib import Path
import random

from datasets import Dataset, DatasetDict, load_from_disk
import pandas as pd
import torch
from transformers import EsmTokenizer
from transformers.utils import logging

from esmtherm.util import write_json, read_json

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default=None, required=True, help='Dataset directory')
parser.add_argument('--csv', type=str, default=None, required=True)
parser.add_argument('--split_csv', type=str, default=None,
                    help='CSV file containing train-val-test splits by wildtype. '
                         'Must contain at least columns "wildtype" and "split"')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test_split', type=float, default=0.1)
parser.add_argument('--n_proc', type=int, default=1)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--tokenize_config_json', type=str, default=None)
parser.add_argument('--sequence_column', type=str, default='aa_seq_full')
parser.add_argument('--label_columns', nargs='+', type=str,
                    default=['deltaG_t', 'log10_K50_t', 'deltaG_c', 'log10_K50_c', 'deltaG'])
args = parser.parse_args()

random.seed(args.seed)


def main():
    # 0. setup
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), dataset_dir / 'args.json')

    try:
        load_from_disk(dataset_dir)
        if not args.overwrite:
            logger.info(f'Found dataset in {dataset_dir}. Skipping...')
            return
    except:
        logger.info(f'No dataset found in {dataset_dir}. Building dataset...')
        pass

    # 1. load and split dataset
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.sequence_column] + args.label_columns)
    df = df.rename({args.sequence_column: 'sequence'}, axis=1)

    split_ratio = {
        'train': 1 - 2 * args.test_split,
        'val': args.test_split,
        'test': args.test_split,
    }
    for k, v in split_ratio.items():
        df[k] = v

    if args.split_csv is not None:
        df_split = pd.read_csv(args.split_csv)
        for split_name in ['train', 'val', 'test']:
            wildtypes = set(df_split.loc[df_split['split'] == split_name, 'WT_name'])
            df.loc[~df['WT_name'].isin(wildtypes), split_name] = 0

    def _split_fxn(row):
        probs = {'train': row['train'], 'val': row['val'], 'test': row['test']}
        if sum(probs.values()) > 0:
            return random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        else:
            return None

    df['split'] = df.apply(_split_fxn, axis=1)
    df.to_csv(dataset_dir / 'dataset.csv', index=False)

    # 2. clean dataset
    cols = ['name', 'sequence', 'WT_name', 'split'] + args.label_columns
    for col in df.columns:
        if col not in cols:
            del df[col]

    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(df.loc[df['split'] == 'train']),
        'val': Dataset.from_pandas(df.loc[df['split'] == 'val']),
        'test': Dataset.from_pandas(df.loc[df['split'] == 'test']),
    })
    dataset_dict = dataset_dict.remove_columns(['__index_level_0__', 'split'])

    # 3. tokenize dataset
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    tokenizer_kwargs = dict(padding='longest')
    if args.tokenize_config_json is not None:
        tokenizer_kwargs.update(
            read_json(args.tokenize_config_json)
        )

    def _tokenize(features):
        features_ = tokenizer(features['sequence'], **tokenizer_kwargs)
        del features_['attention_mask']
        return features_

    dataset_dict = dataset_dict.map(_tokenize, batched=False, num_proc=min(torch.get_num_threads(), args.n_proc))

    # 4. save dataset
    dataset_dict.save_to_disk(dataset_dir)


if __name__ == '__main__':
    main()
