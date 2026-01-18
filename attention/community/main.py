import multiprocessing
import os,argparse

import numpy as np
import pandas as pd
from generate_esm2_attns import embed_seqs, load_model, embed_single_seq
from utils import read_fasta, read_fasta_folder, DiscardHead
from get_key_residues import extract_key_residues, extract_keyres_from_single_seq
from attn_to_dictionary import (extract_words, single_seq_community_detection, single_seq_wordlist,
                                dictionary_raw_counts, restype_freq_baseline, dictionary_normalized_counts)
import multiprocessing as mlp

def single_seq_workflow(name, seq, model, alphabet, batch_converter, device, head_cls_dict):
    attentions = embed_single_seq(seq, model, alphabet, batch_converter, device)
    print('Finished attention extraction: %s' % name)
    ss3 = None
    df_keyres, excluded_heads = extract_keyres_from_single_seq(seq, name, attentions, ss3, head_cls_dict, max_cutoff=10)
    manual_discard_heads = DiscardHead(head_cls_dict)
    ignored_heads = np.logical_and(manual_discard_heads, excluded_heads)
    branches_louvain = single_seq_community_detection(attentions, ignored_heads)
    df_wordlist = single_seq_wordlist(name, seq, attentions, branches_louvain, ignored_heads)
    print('Finished wordlist generation: %s' % name)
    return df_keyres, df_wordlist


def main():
    parser = argparse.ArgumentParser()
    WORKDIR = os.path.abspath(os.path.join('.')) + '/'
    max_length = 1024
    n_processes = 4
    print(WORKDIR)
    parser.add_argument(
        "--fasta_name", type=str, required=True,
        help="Path of fasta files")

    parser.add_argument(
        "--mode", type=str,required=True,
        help="Specific mode to run: single or batch")

    parser.add_argument(
        "--ss3_file", type=str,
        help="Path of ss3 file, containing secondary structures of input sequences")

    '''parser.add_argument(
        "--tmp_dir", type=str, default=os.path.join(WORKDIR, 'tmp/'),
        help="Path of temporary files directory")'''

    parser.add_argument(
        "--out_dir", type=str, default=os.path.join(WORKDIR, 'outputs/'),
        help="Path of output attention directory")

    parser.add_argument(
        "--ignored_head_file", type=str, default=os.path.join(WORKDIR, 'data/head_cls_dict.pkl'),
        help="Path of output attention directory")

    args = parser.parse_args()
    # Define the name of the dataset by fasta file or path name
    if args.fasta_name.endswith('.fasta'):
        seq_names, seqs = read_fasta(args.fasta_name)
        base_name = os.path.basename(args.fasta_name).replace('.fasta', '')
    elif os.path.basename(args.fasta_name) != '':
        seq_names, seqs = read_fasta(args.fasta_name)
        base_name = os.path.basename(args.fasta_name).split('.')[0]
    else:
        seq_names, seqs = read_fasta_folder(args.fasta_name)
        base_name = os.path.dirname(args.fasta_name).split('/')[-1]
    print(base_name, seq_names)

    # os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "single":
        print('Input fasta name: {}'.format(args.fasta_name))
        print(f'Input sequence: {seqs[0]}\nSequence length: {len(seqs[0])}')
        if len(seqs) != 1:
            raise ValueError('Single mode not supported')
    else:
        print('Input fasta name: {}'.format(args.fasta_name))
        print(f'Containing %s sequences' % (len(seqs)))
        if len(seqs) <= 1:
            raise ValueError('Batch mode not supported')
    all_seq_dict = {k: v for k, v in zip(seq_names, seqs) if len(v) <= max_length}
    print(all_seq_dict.keys())
    print(args.ss3_file)
    output_path = args.out_dir
    # Step 1: Run ESM2 to retrieve attention matrices
    model, alphabet, batch_converter, device = load_model()
    # Read head classification dict and determine which heads are to be ignored
    head_cls_dict = pd.read_pickle(args.ignored_head_file)
    list_pool_results = []
    pool = mlp.Pool(processes=n_processes)
    for i, (name, seq) in enumerate(all_seq_dict.items()):

        pool_result = pool.apply_async(single_seq_workflow,
                                    (name, seq, model, alphabet, batch_converter, device, head_cls_dict,))
        list_pool_results.append(pool_result)

    pool.close()
    pool.join()
    list_df_keyres_all = [k.get()[0] for k in list_pool_results]
    list_df_wordlist_all = [k.get()[1] for k in list_pool_results]

    df_keyres_all_raw = pd.concat(list_df_keyres_all)
    df_keyres_all_raw.to_csv(os.path.join(output_path, f'{base_name}_keyres.csv'))
    df_wordlist_all_raw = pd.concat(list_df_wordlist_all)
    df_wordlist_all_raw.to_csv(os.path.join(output_path, f'{base_name}_dict_raw.csv'))

    dictionary, word_dataframe_bycount = dictionary_raw_counts(df_wordlist_all_raw)

    # Save normalized dictionary and sequence segmentation table
    np.save(os.path.join(output_path, f'{base_name}_dictionary_rawcount.npy'), dictionary)
    word_dataframe_bycount.to_csv(os.path.join(output_path, f'{base_name}_segment_table_rawcount.csv'))

    # Step 4: Create word dictionary using normalized count
    res_prob_baseline = restype_freq_baseline(all_seq_dict)
    dictionary_normalized, word_dataframe_normalized = dictionary_normalized_counts(df_wordlist_all_raw, res_prob_baseline)

    # Save normalized dictionary and sequence segmentation table
    np.save(os.path.join(output_path, f'{base_name}_dictionary_normalized.npy'), dictionary_normalized)
    word_dataframe_normalized.to_csv(os.path.join(output_path, f'{base_name}_segment_table_normalized.csv'))

    return df_keyres_all_raw, df_wordlist_all_raw
if __name__ == "__main__":
    main()



'''  
============================Requirements===============================================
conda create --name pt3.9 python=3.9
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (change cuda version)
Biopython==1.79
pandas==1.3.5
fair-esm==2.0
networkx==3.2.1
python==3.9.19
scipy==1.11.4
'''

'''USAGE'''
'''for single mode'''

'''
python main.py --fasta_name ../tmp/single.fasta --mode single --ss3_file ../tmp/single.pkl
'''

'''for batch mode'''

'''
python main.py --fasta_name ../tmp/batch.fasta --mode batch --ss3_file ../tmp/batch.pkl
'''