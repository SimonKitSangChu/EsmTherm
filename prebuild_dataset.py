import argparse
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.utils import logging

from esmtherm.data import to_wildtype, to_mutant
from esmtherm.util import write_json, read_json

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='data/K50_dG_Dataset1_Dataset2.csv')
parser.add_argument('--output_dir', type=str, default='datasets/analysis', help='Output directory')
parser.add_argument('--wildtype_map', type=str, default=None)
parser.add_argument('--dG_names', nargs='+', type=str, default=['deltaG_t', 'deltaG_c', 'deltaG'],
                    choices=['deltaG_t', 'deltaG_c', 'deltaG'])
parser.add_argument('--K50_names', nargs='+', type=str, default=['log10_K50_t', 'log10_K50_c'],
                    choices=['log10_K50_t', 'log10_K50_c'])
parser.add_argument('--max_deltaG_std', type=float, default=2,
                    help='Max standard deviation of deltaG_t allowed in duplicate data')
parser.add_argument('--max_log_K50_std', type=float, default=0.5,
                    help='Max standard deviation of log10_K50_t allowed in duplicate data')
parser.add_argument('--max_log_K50_95CI', type=float, default=0.5,
                    help='Max 95% CI of log10_K50 in fitting')
args = parser.parse_args()

mpl.style.use('seaborn-v0_8-colorblind')


def main():
    # 0. setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(vars(args), output_dir / 'args.json')

    # 1. load data
    df = pd.read_csv(args.csv)

    if args.wildtype_map is None:
        # beware of double mutant since their "wildtype" is the single mutant
        # fix WT_name by splitting with .pdb and mut_type by matching WT_name == name

        if 'WT_name' in df.columns:
            df['WT_name_old'] = df['WT_name']
            df['WT_name'] = df['WT_name'].apply(lambda x: x.split('.pdb')[0] + '.pdb' if '.pdb' in x else x)
        else:
            df['WT_name'] = df['name'].apply(to_wildtype)
        if 'mut_type' in df.columns:
            df['mut_type_old'] = df['mut_type']
            df['mut_type'] = df.apply(
                lambda row: 'wt' if row['WT_name'] == row['name'] else to_mutant(row['name']),
                axis=1,
            )
    else:
        # map by wildtype_map instead of replying on dataset entries
        wildtype_map = read_json(args.wildtype_map)
        df['WT_name'] = df['aa_seq_full'].map(wildtype_map)
        df['mut_type'] = df['aa_seq_full']

    df_raw = df.copy()

    dG_names = ['deltaG_t', 'deltaG_c', 'deltaG']
    K50_names = ['log10_K50_t', 'log10_K50_c']

    # 2. visualize data
    # 2.1 error analysis on duplicate (full) sequences
    if args.max_deltaG_std is None and args.max_log_K50_std is None:
        logger.info('No filtering criteria specified. Use all data.')
        df_filtered = None
    else:
        # 2.1.1 save filtered data
        df_filtered = df.copy()

        # filtering by discrepancy between duplicates
        if args.max_deltaG_std is not None:
            criteria = df.groupby('aa_seq_full')[dG_names].std().fillna(0) < args.max_deltaG_std
            sequences_ = criteria[criteria].index
            df_filtered = df_filtered[df_filtered['aa_seq_full'].isin(sequences_)]
        if args.max_log_K50_std is not None:
            criteria = df.groupby('aa_seq_full')[K50_names].std().fillna(0) < args.max_log_K50_std
            sequences_ = criteria[criteria].index
            df_filtered = df_filtered[df_filtered['aa_seq_full'].isin(sequences_)]

        # filtering by 95% CI of log10_K50
        if args.max_log_K50_95CI is not None:
            df_filtered = df_filtered[df_filtered['log10_K50_t'] < args.max_log_K50_95CI]
            df_filtered = df_filtered[df_filtered['log10_K50_c'] < args.max_log_K50_95CI]

        df_filtered.to_csv(output_dir / 'filtered_data.csv', index=False)

        # 2.1.2 visualize filtered data with error bars
        df_std = df_filtered.groupby('aa_seq_full')[dG_names].std().dropna()
        df_mean = df_filtered.groupby('aa_seq_full')[dG_names].mean()
        df_sta = df_std.join(
            df_mean,
            lsuffix='_std', rsuffix='_mean'
        )

        df_std = df.groupby('aa_seq_full')[K50_names].std().dropna()
        df_mean = df.groupby('aa_seq_full')[K50_names].mean()
        df_sta = df_sta.join(
            df_std.join(
                df_mean,
                lsuffix='_std', rsuffix='_mean'
            ),
        )

        for dG_name, K50_name in (('deltaG_t', 'log10_K50_t'), ('deltaG_c', 'log10_K50_c')):
            dG_mean = f'{dG_name}_mean'
            dG_std = f'{dG_name}_std'
            K50_mean = f'{K50_name}_mean'
            K50_std = f'{K50_name}_std'

            jg = sns.jointplot(
                data=df_sta, x=dG_mean, y=K50_mean,
                s=0.2, alpha=0.2,
                # kind='reg', truncate=False,
                # scatter_kws=dict(s=0.2, alpha=0.2),
            )
            jg.ax_joint.errorbar(
                x=df_sta[dG_mean], y=df_sta[K50_mean],
                xerr=df_sta[dG_std], yerr=df_sta[K50_std],
                fmt='none', ecolor='gray', alpha=0.1,
                elinewidth=0.1, errorevery=5,
            )

        jg.ax_joint.set_xlabel(dG_name)
        jg.ax_joint.set_ylabel(K50_name)
        jg.fig.suptitle('Error analysis on duplicate (full) sequences')
        plt.savefig(output_dir / f'{dG_name}_{K50_name}_filtered.png')
        plt.close()

    # 2.2 visualize dataset-level statistics
    # 2.2.0 clean data
    df_unique = df.drop_duplicates(subset=['aa_seq_full'], keep=False)

    # 2.2.1 duplications
    fig, axes = plt.subplots(1+(df_filtered is not None), 1, tight_layout=True, dpi=300)
    axes[0].pie(
        [df.shape[0]-df_unique.shape[0], df_unique.shape[0]],
        labels=['other (exclusive)', 'unique'], autopct='%1.1f%%',
        explode=[0, 0.1]
    )
    axes[0].set_title('Sequences')

    if df_filtered is not None:
        axes[1].pie(
            [df.shape[0]-df_filtered.shape[0], df_filtered.shape[0]],
            labels=['other (exclusive)', 'filtered'], autopct='%1.1f%%',
            explode=[0, 0.1]
        )
        axes[1].set_title('Filtered')

    plt.savefig(output_dir / 'duplicates.png')
    plt.close()

    # 2.2.3 distribution of sequence length
    fig, axes = plt.subplots(1+(df_filtered is not None), 1, tight_layout=True, dpi=300)
    if df_filtered is None:
        axes = [axes]

    df['aa_seq_full'].str.len().hist(label='all', log=True, ax=axes[0])
    if df_filtered is None:
        df_unique['aa_seq_full'].str.len().hist(label='unique', log=True, ax=axes[0])
    else:
        df_filtered['aa_seq_full'].str.len().hist(label='filtered', log=True, ax=axes[0])
    axes[0].set_title('Full sequence length')

    if 'aa_seq' in df.columns:
        df['aa_seq'].str.len().hist(label='all', log=True, ax=axes[1])
        if df_filtered is None:
            df_unique['aa_seq'].str.len().hist(label='unique', log=True, ax=axes[1])
        else:
            df_filtered['aa_seq'].str.len().hist(label='filtered', log=True, ax=axes[1])
        axes[1].set_title('Sequence length')

    plt.legend()
    plt.savefig(output_dir / 'sequence_length.png')
    plt.close()

    # 2.2.3 agreement between trypsin and chymotrypsin
    fig, axes = plt.subplots(1+(df_filtered is not None), 1, tight_layout=True, dpi=300)
    if df_filtered is None:
        axes = [axes]

    df.plot.scatter(x='deltaG_t', y='deltaG_c', s=0.2, alpha=0.2, ax=axes[0])
    lims = [
        min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
        max(axes[0].get_xlim()[1], axes[0].get_ylim()[1]),
    ]
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_aspect('equal')
    axes[0].set_title('All data')

    print(
        'Correlation between trypsin and chymotrypsin:',
        df['deltaG_t'].corr(df['deltaG_c'], method='spearman')
    )

    if df_filtered is not None:
        df_filtered.plot.scatter(x='deltaG_t', y='deltaG_c', s=0.2, alpha=0.2, ax=axes[1])
        lims = [
            min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
            max(axes[1].get_xlim()[1], axes[1].get_ylim()[1]),
        ]
        axes[1].set_xlim(lims)
        axes[1].set_ylim(lims)
        axes[1].set_aspect('equal')
        axes[1].set_title('Filtered data')

        print(
            'Correlation between trypsin and chymotrypsin: (filtered)',
            df_filtered['deltaG_t'].corr(df_filtered['deltaG_c'], method='spearman')
        )

    axes[0].set_title('Agreement between trypsin and chymotrypsin')
    plt.savefig(output_dir / 'trypsin_chymotrypsin_agreement.png')
    plt.close()

    # 2.2.4 agreement between aa_seq and aa_seq_full (impact of linkers)
    if 'aa_seq' in df.columns:
        dG_name = 'deltaG'
        K50_name = 'log10_K50'
        df[K50_name] = (df['log10_K50_t'] + df['log10_K50_c']) / 2  # ad hoc solution

        group = df.groupby('aa_seq')[dG_names + K50_names + [K50_name]]
        df_sta = group.agg(['mean', 'std']).swaplevel(0, 1, axis=1)

        ax = sns.jointplot(
            data=df_sta['mean'], x=dG_name, y=K50_name,
            s=0.2, alpha=0.2,
            # kind='reg', truncate=False,
            # scatter_kws=dict(s=0.2, alpha=0.2),
        )
        ax.ax_joint.errorbar(
            x=df_sta['mean'][dG_name], y=df_sta['mean'][K50_name],
            xerr=df_sta['std'][dG_name], yerr=df_sta['std'][K50_name],
            fmt='none', ecolor='gray', alpha=0.1,
            elinewidth=0.1, errorevery=5,
        )

        ax.ax_joint.set_xlabel(dG_name)
        ax.ax_joint.set_ylabel(K50_name)
        ax.fig.suptitle('Error analysis on duplicate sequences (without linker)')
        plt.tight_layout()
        plt.savefig(output_dir / f'{dG_name}_{K50_name}_linker.png')
        plt.close()

        if df_filtered is not None:
            df_filtered[K50_name] = (df_filtered['log10_K50_t'] + df_filtered['log10_K50_c']) / 2  # ad hoc solution
            group = df_filtered.groupby('aa_seq')[dG_names + K50_names + [K50_name]]
            df_sta = group.agg(['mean', 'std']).swaplevel(0, 1, axis=1)

            ax = sns.jointplot(
                data=df_sta['mean'], x=dG_name, y=K50_name,
                s=0.2, alpha=0.2,
                # kind='reg', truncate=False,
                # scatter_kws=dict(s=0.2, alpha=0.2),
            )
            ax.ax_joint.errorbar(
                x=df_sta['mean'][dG_name], y=df_sta['mean'][K50_name],
                xerr=df_sta['std'][dG_name], yerr=df_sta['std'][K50_name],
                fmt='none', ecolor='gray', alpha=0.1,
                elinewidth=0.1, errorevery=5,
            )

            ax.ax_joint.set_xlabel(dG_name)
            ax.ax_joint.set_ylabel(K50_name)
            ax.fig.suptitle('Error analysis on duplicate sequences (without linker)')
            plt.tight_layout()
            plt.savefig(output_dir / f'{dG_name}_{K50_name}_linker_filtered.png')
            plt.close()

    # 2.3 visualize wildtype-level statistics
    # 2.3.1 distribution of number of mutants per wildtype
    counts_full = df.groupby('WT_name')['mut_type'].count()
    counts_unique = df_unique.groupby('WT_name')['mut_type'].count()

    kwargs = {
        'range': (counts_full.min(), counts_full.max()),
        'bins': 20,
        # 'alpha': 0.25,
        'log': True
    }

    ax = counts_full.hist(label='all', **kwargs)
    if df_filtered is None:
        ax = counts_unique.hist(label='unique', ax=ax, **kwargs)
    else:
        counts_filtered = df_filtered.groupby('WT_name')['mut_type'].count()
        ax = counts_filtered.hist(label='filtered', ax=ax, **kwargs)

    ax.set_title('Distribution of number of mutants per wildtype')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'num_mutants_per_wildtype.png')
    plt.close()

    if args.wildtype_map is None:
        counts_full = counts_full.sort_values(ascending=False)
        counts_full.to_csv(output_dir / 'num_mutants_per_wildtype.csv')
    else:
        counts_filtered = counts_filtered.sort_values(ascending=False)
        counts_filtered.to_csv(output_dir / 'num_mutants_per_wildtype.csv')


if __name__ == '__main__':
    main()
