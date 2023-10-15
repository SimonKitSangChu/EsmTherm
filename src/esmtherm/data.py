from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


AA_CODES = 'ACDEFGHIKLMNPQRSTVWY'


class MutationError(Exception):
    pass


def is_single(mut: str, keep_ins: bool = False, keep_del: bool = False, wt_sequence: Optional[str] = None) -> bool:
    try:
        # deal with ins and del
        if (mut.startswith('ins') and keep_ins) or (mut.startswith('del') and keep_del):
            assert mut[3] in AA_CODES
            resid = int(mut[4:])
            if wt_sequence is not None:
                assert wt_sequence[resid - 1] == mut[3]
            return True

        # deal with point mutation
        if mut[0] in AA_CODES and mut[-1] in AA_CODES:
            resid = int(mut[1:-1])
            if wt_sequence is not None:
                assert wt_sequence[resid - 1] == mut[0]
            return True

    except (ValueError, AssertionError):
        return False

    return False


def is_regular(name: str) -> bool:
    if '_' not in name:
        return True

    return len(name.split('_')) == 2


def to_wildtype(name: str) -> str:
    return name.split('_')[0].replace('.pdb', '')


def to_mutant(name: str) -> Optional[str]:
    if '_' not in name:
        return
    return '_'.join(name.split('_')[1:])


def name2mut(name: str, wt_name: Optional[str] = None) -> str:
    if wt_name is None:
        mut = name
    else:
        mut = name.replace(wt_name + '_', '')

    if '_' in mut:
        mut = mut[::-1].split('_')[0][::-1]  # take from last _ to end

    return mut


def classify_mut(mut: str, wt_name: Optional[str] = None) -> str:
    if mut.startswith('wt') or (mut == wt_name):
        return 'wt'
    if is_single(mut, keep_ins=True, keep_del=True):
        if mut.startswith('ins'):
            return 'ins'
        if mut.startswith('del'):
            return 'del'
        return 'single'
    return 'others'


def mutate(sequence: str, mut: str, raise_error: bool = True) -> str:
    try:
        # deal with ins and del
        if mut.startswith('ins'):
            assert mut[3] in AA_CODES
            return sequence[:int(mut[4:]) - 1] + mut[3] + sequence[int(mut[4:]) - 1:]
        if mut.startswith('del'):
            assert mut[3] in AA_CODES
            return sequence[:int(mut[4:]) - 1] + sequence[int(mut[4:]) - 1 + 1:]

        # deal with point mutation
        wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
        assert sequence[pos - 1] == wt_aa
        return sequence[:pos - 1] + mut_aa + sequence[pos:]
    except (ValueError, AssertionError):
        if not raise_error:
            return None
        pos = int(mut[1:-1])
        raise MutationError(f'Not supported mutation: {mut} at position {sequence[pos-1]}{pos}')


@dataclass
class EsmDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self._pad(features)
        if 'wildtype_input_ids' in features[0]:
            wildtype_batch = self._pad([
                {k.replace('wildtype_', ''): v for k, v in feature.items() if k.startswith('wildtype_')}
                for feature in features
            ])
            batch.update({f'wildtype_{k}': v for k, v in wildtype_batch.items()})

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

    def _pad(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
