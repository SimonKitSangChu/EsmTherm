from os import PathLike
import json
import hashlib
from typing import Dict, Any, Optional, Union, NewType, Iterable, Generator

from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy import stats


Sequence = NewType('Sequence', Union[str, SeqRecord, Seq])
Sequences = NewType('Sequences', Union[Dict[str, Sequence], Iterable[Sequence]])


def read_json(json_file: PathLike) -> Dict[str, Any]:
    with open(json_file) as handle:
        return json.load(handle)


def write_json(data: Dict[str, Any], json_file: PathLike) -> None:
    with open(json_file, 'w') as handle:
        json.dump(data, handle, indent=2)


def get_hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


def str2seqrecord(seq: str, name: Optional[str] = None) -> SeqRecord:
    if name is None:
        name = get_hash_string(seq)
    return SeqRecord(Seq(seq), id=name, name='', description='')


def get_record(sequence: Sequence) -> Sequence:
    if isinstance(sequence, str):
        return str2seqrecord(sequence)
    elif isinstance(sequence, SeqRecord):
        return sequence
    elif isinstance(sequence, Seq):
        return str2seqrecord(str(sequence))
    else:
        raise TypeError(f'Unknown type {type(sequence)}')


def get_records(sequences: Sequences) -> Union[Sequences, Generator[Sequence, None, None]]:
    if isinstance(sequences, dict):
        return {name: get_record(seq) for name, seq in sequences.items()}
    elif isinstance(sequences, SeqRecord) or isinstance(sequences, Seq) or isinstance(sequences, str):
        return [get_record(sequences)]
    else:
        return (get_record(seq) for seq in sequences)


def write_fasta(sequences: Union[Sequence, Sequences], fasta: PathLike) -> None:
    sequences = get_records(sequences)
    SeqIO.write(sequences, fasta, 'fasta')


def read_fasta(fasta: PathLike, format: str = 'record') -> Dict[str, Sequence]:
    sequences = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))
    if format == 'record':
        return sequences
    elif format == 'str':
        return {name: str(seq.seq) for name, seq in sequences.items()}
    else:
        raise ValueError(f'Only support format "record" or "str", got {format}')


def read_mmseqs_clusters(fasta: PathLike, tsv: PathLike) -> Dict[str, str]:
    clusters = {}
    records = read_fasta(fasta, 'str')

    with open(tsv, 'r') as handle:
        for line in handle:
            parent, child = line.strip().split('\t')
            clusters[records[child]] = records[parent]

    return clusters


def pearsonr(x: Iterable[float], y: Iterable[float]):
    try:
        return stats.pearsonr(x, y)
    except ValueError:
        return (float('nan'), float('nan'))


def spearmanr(x: Iterable[float], y: Iterable[float]):
    try:
        return stats.spearmanr(x, y)
    except ValueError:
        return (float('nan'), float('nan'))


def extract_bfactors(pdb_file: PathLike, **kwargs) -> Union[Dict[str, list[float]], list[float]]:
    parser = PDBParser(QUIET=True, **kwargs)
    structure = parser.get_structure('PDB', pdb_file)

    b_factors_dict = {}
    for model in structure:
        for chain in model:
            b_factors_dict[chain.id] = []
            for residue in chain:
                if 'C' in residue:
                    atom = residue['C']
                    b_factors_dict[chain.id].append(atom.get_bfactor())

    if len(b_factors_dict) == 1:
        return next(iter(b_factors_dict.values()))
    else:
        return b_factors_dict

