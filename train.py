import argparse
from pathlib import Path
import shutil

from datasets import load_from_disk, DatasetDict, concatenate_datasets
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmForSequenceClassification, TrainingArguments, EarlyStoppingCallback,\
    Trainer, get_constant_schedule, EsmConfig
from transformers.utils import logging

from esmtherm.util import write_json
from esmtherm.data import EsmDataCollatorWithPadding
from esmtherm.model import EsmPooledClassification

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default=None, required=True)
parser.add_argument('--output_dir', type=str, default='output_dir')
parser.add_argument('--model_name', type=str, default='facebook/esm2_t6_8M_UR50D', choices=[
    'facebook/esm2_t6_8M_UR50D',
    'facebook/esm2_t12_35M_UR50D',
    'facebook/esm2_t30_150M_UR50D',
    'facebook/esm2_t33_650M_UR50D',
    'facebook/esm2_t36_3B_UR50D',
    'facebook/esm2_t48_15B_UR50D',
    'facebook/esm1b_t33_650M_UR50S',
    'facebook/esm1v_t33_650M_UR90S_1',
])
parser.add_argument('--label_column', type=str, default='deltaG',
                    help='Label column. Typical options are deltaG_t, log10_K50_t')
parser.add_argument('--WT_names', type=str, nargs='+', default=None,
                    help='WT names. If not specified, use all WT_name in the dataset')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--global_batch_size', type=int, default=2048)
parser.add_argument('--local_batch_size', type=int, default=32)
parser.add_argument('--logging_steps', default=5, type=int, help='Logging steps')
parser.add_argument('--save_steps', default=20, type=int, help='Save steps')
parser.add_argument('--no_cuda', action='store_true', help='No CUDA mode')
parser.add_argument('--fp16', action='store_true', help='Use fp16')
parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='Resume from training checkpoint')
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_csv', type=str, default=None)
parser.add_argument('--train_from_scratch', action='store_true')
parser.add_argument('--feature_extraction', action='store_true')
parser.add_argument('--overwrite', action='store_true')
args, unk = parser.parse_known_args()


def main():
    # 0. setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), output_dir / 'args.json')

    if unk:
        logger.warning(f'unknown arguments {unk}')
        write_json(unk, output_dir / 'unk.json')

    # 1. setup dataset
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    dataset_dict_org = load_from_disk(args.dataset_dir)

    # move not-selected WT to test set
    if args.WT_names is not None:
        wt_names = set(args.WT_names)

        def resplit(row):
            if row['WT_name'] in wt_names:
                return {'keep': True}
            return {'keep': False}

        dataset_dict_org = dataset_dict_org.map(resplit, num_proc=4)
        dataset_dict_org = DatasetDict({
            'train': dataset_dict_org['train'].filter(lambda x: x['keep']),
            'val': dataset_dict_org['val'].filter(lambda x: x['keep']),
            'test': concatenate_datasets([
                dataset_dict_org['train'].filter(lambda x: not x['keep']),
                dataset_dict_org['val'].filter(lambda x: not x['keep']),
                dataset_dict_org['test']
            ])
        })

    dataset_dict = dataset_dict_org.rename_column(args.label_column, 'labels')
    allowed_columns = ['input_ids', 'attention_mask', 'labels']
    dataset_dict = dataset_dict.remove_columns(
        [col for col in dataset_dict['train'].column_names if col not in allowed_columns]
    )

    # 2. set up training config
    model_cls = EsmForSequenceClassification if not args.feature_extraction else EsmPooledClassification
    best_ckpt_dir = output_dir / 'checkpoint-best'

    if not best_ckpt_dir.exists() or args.overwrite:
        # 2.1 training arguments
        if args.debug:
            local_batch_size = 1
            global_batch_size = torch.cuda.device_count() * local_batch_size if torch.cuda.is_available() \
                else local_batch_size
        else:
            local_batch_size = args.local_batch_size
            global_batch_size = args.global_batch_size

        grad_steps = global_batch_size // local_batch_size
        if torch.cuda.is_available():
            grad_steps = grad_steps // torch.cuda.device_count()

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            logging_strategy='steps',
            evaluation_strategy='steps',
            save_strategy='steps',
            logging_steps=args.logging_steps,
            eval_steps=args.save_steps,
            save_steps=args.save_steps,
            eval_accumulation_steps=1,
            save_total_limit=1,
            num_train_epochs=1 if args.debug else args.max_epochs,
            per_device_train_batch_size=local_batch_size,
            per_device_eval_batch_size=local_batch_size,
            gradient_accumulation_steps=grad_steps,
            dataloader_num_workers=torch.cuda.device_count() if torch.cuda.is_available() else 1,
            disable_tqdm=False,
            load_best_model_at_end=True,  # early stopping callback
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            seed=args.seed,
            learning_rate=args.lr,
            lr_scheduler_type='constant',
            no_cuda=args.no_cuda,
            fp16=args.fp16,
            prediction_loss_only=True,
            do_eval=True,
            ddp_find_unused_parameters=True,
        )

        # 2.2 trainer config
        if args.train_from_scratch:
            model_config = EsmConfig.from_pretrained(args.model_name)
            model_config.num_labels = 1
            model = model_cls(model_config)
        else:
            model = model_cls.from_pretrained(args.model_name, num_labels=1)  # regression

        optimizer = torch.optim.AdamW(
            model.classifier.parameters() if args.feature_extraction else model.parameters(),
            lr=args.lr
        )
        scheduler = get_constant_schedule(optimizer)

        trainer = Trainer(
            model=model,
            data_collator=EsmDataCollatorWithPadding(tokenizer=tokenizer, padding=True),
            args=training_args,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict['val'],
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.patience, early_stopping_threshold=0.01)
            ],
            optimizers=(optimizer, scheduler)
        )

        # 3. train and save model
        if args.resume_from_checkpoint is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

        # remove all checkpoints except the best one
        for ckpt in output_dir.glob('checkpoint-*'):
            if ckpt != best_ckpt_dir:
                shutil.rmtree(ckpt)

        trainer.model.save_pretrained(best_ckpt_dir)

    # 4. evaluate on all samples
    if args.output_csv is None:
        return

    model = model_cls.from_pretrained(best_ckpt_dir)
    model.eval()
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    def _evaluate(rows):
        rows = [{key: rows[key][i] for key in rows} for i in range(len(rows[list(rows.keys())[0]]))]

        with torch.no_grad():
            collator = EsmDataCollatorWithPadding(tokenizer=tokenizer, padding=True)
            features = collator(rows)
            input_ids = features['input_ids'].to(model.device)
            outputs = model(input_ids=input_ids)

        return {'prediction': outputs.logits.tolist()}

    dataset_dict = dataset_dict.map(
        _evaluate,
        batched=True,
        batch_size=args.local_batch_size,
        desc='Evaluating',
    )

    # 5. save predictions
    df = []
    for split_name, dataset in dataset_dict.items():
        df_ = dataset.to_pandas()[['labels', 'prediction']]
        df_['prediction'] = df_['prediction'].apply(lambda x: x[0])

        cols = [
            col for col in dataset_dict_org[split_name].column_names
            if col not in df_.columns and col not in ('input_ids',)
        ]
        df_org = dataset_dict_org[split_name].to_pandas()[cols]
        df_['split'] = split_name

        df_ = df_.join(df_org)
        df.append(df_)

    df = pd.concat(df, ignore_index=True)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
