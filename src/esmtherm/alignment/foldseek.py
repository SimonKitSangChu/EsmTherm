from os import PathLike
from pathlib import Path
import subprocess as sp
from typing import Any, Dict, Iterable, Optional

from Bio.SeqRecord import SeqRecord
import pandas as pd
from transformers.utils import logging

from esmtherm.util import write_fasta

logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def kwargs2flags(kwargs: Dict[str, Any]) -> str:
    if not kwargs:
        return ''
    return ' '.join([f'--{k} {v}' for k, v in kwargs.items()])


def create_db(fasta: PathLike, db: PathLike, **kwargs) -> None:
    cmd = f'foldseek createdb {fasta} {db}' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True)


def search(queryDB: PathLike, targetDB: PathLike, resultDB: PathLike, clean_tmp: bool = True, **kwargs) -> None:
    cmd = f'foldseek search {queryDB} {targetDB} {resultDB} tmp' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True)
    if clean_tmp:
        sp.run('rm -rf tmp', shell=True, check=True)


def alignall(queryDB: PathLike, targetDB: PathLike, resultDB: PathLike = 'alignall', **kwargs) -> None:
    # (How to create a fake prefiltering for all-vs-all alignments)
    # reference: https://github.com/soedinglab/mmseqs2/wiki
    queryDB = Path(queryDB).resolve()
    targetDB = Path(targetDB).resolve()
    resultDB = Path(resultDB).resolve()

    # check if all DBs are in the same directory
    assert queryDB.parent == targetDB.parent == resultDB.parent, 'All DB must be in the same directory!'
    dirname = queryDB.parent
    queryDB, targetDB, resultDB = queryDB.name, targetDB.name, resultDB.name

    cmd = f'''
    cd {dirname};
    ln -s {targetDB}.index {resultDB}_pref;
    INDEX_SIZE="$(echo $(wc -c < "{targetDB}.index"))";
    awk -v size=$INDEX_SIZE '{{ print $1"\t0\t"size; }}' "{queryDB}.index" > "{resultDB}_pref.index";
    awk 'BEGIN {{ printf("%c%c%c%c",7,0,0,0); exit; }}' > "{resultDB}_pref.dbtype";
    foldseek align "{queryDB}" "{targetDB}" "{resultDB}_pref" "{resultDB}" -a {kwargs2flags(kwargs)};
    '''
    sp.run(cmd, shell=True, check=True)


def convertalis(queryDB: PathLike, targetDB: PathLike, resultDB: PathLike, result: PathLike, **kwargs) -> None:
    kwargs['format-output'] = kwargs.get(
        'format-output', 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,'
                         'prob,lddt,alntmscore'
    )

    cmd = f'foldseek convertalis {queryDB} {targetDB} {resultDB} {result}.m8 ' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True)


def parse_m8(m8: PathLike, format_columns: Optional[Iterable] = None) -> pd.DataFrame:
    if format_columns is None:
        format_columns = [
            'query', 'target', 'pident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue',
            'bits', 'prob', 'lddt', 'alntmscore'
        ]

    df = pd.read_csv(m8, sep='\t', header=None)
    df.columns = format_columns
    return df


def pairwise_align(
        query_dir: PathLike,
        target_dir: Optional[PathLike] = None,
        query_db: str = 'queryDB',
        target_db: str = 'targetDB',
        result_db: str = 'resultDB',
        result: str = 'result',
        all_to_all: bool = True,
        drop_self: bool = False,
) -> pd.DataFrame:
    create_db(query_dir, query_db)

    if target_dir is None:
        target_db = query_db
    else:
        create_db(target_db, target_db)

    if all_to_all:
        alignall(query_db, target_db, result_db)
    else:
        search(query_db, target_db, result_db)

    convertalis(query_db, target_db, result_db, result)
    df = parse_m8(str(result) + '.m8')

    if drop_self:
        df = df[df['query'] != df['target']]

    return df
