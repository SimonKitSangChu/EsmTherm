import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmForSequenceClassification, EsmForMaskedLM
from transformers.utils import logging
from tqdm import tqdm

from esmtherm.data import is_single

tqdm.pandas()

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default=None, required=True)
parser.add_argument('--input_csv', type=str, default=None, required=True)
parser.add_argument('--output_csv', type=str, default=None, required=True)
parser.add_argument('--mut_column', type=str, default='mut', help='Mutation column name')
parser.add_argument('--seq_column', type=str, default='sequence', help='Sequence column name')
parser.add_argument('--unsupervised', action='store_true')
args, unk = parser.parse_known_args()


def main():
    # 0. setup
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')

    if args.unsupervised:
        model = EsmForMaskedLM.from_pretrained(args.model_name_or_path)

        def _evaluate(row):
            # return None for non-point mutations
            mut = row[args.mut_column]
            sequence = row[args.seq_column]
            if not is_single(mut) or sequence is None:
                return

            # masked marginal scoring
            aa_wt, resid, aa_mt = mut[0], int(mut[1:-1]), mut[-1]
            idx = resid - 1
            assert sequence[idx] == aa_mt  # (mutant) sequence check

            input_ids = tokenizer.encode(sequence)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
            input_ids[0, idx] = tokenizer.mask_token_id

            with torch.no_grad():
                outputs = model(input_ids=input_ids)

            score_mt = outputs.logits[0, idx, tokenizer.convert_tokens_to_ids(aa_mt)].item()
            score_wt = outputs.logits[0, idx, tokenizer.convert_tokens_to_ids(aa_wt)].item()
            score = score_mt - score_wt
            return score

    else:
        model = EsmForSequenceClassification.from_pretrained(args.model_name_or_path)

        def _evaluate(row):
            sequence = row[args.seq_column]
            if not sequence:
                return

            input_ids = tokenizer.encode(sequence)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            return outputs.logits[0, 0].item()

    model.eval()
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    df = pd.read_csv(input_csv)
    df[args.seq_column] = df[args.seq_column].fillna(value='')
    df['prediction'] = df.progress_apply(_evaluate, axis=1)
    df.to_csv(output_csv, index=None)


if __name__ == '__main__':
    main()
