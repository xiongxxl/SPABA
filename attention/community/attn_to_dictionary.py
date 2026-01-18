import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
import networkx as nx
from utils import ReadFromTar, ResidueClassification, TimeOut

# final size of final dictionary (both by raw and normalized counts)
DICT_SIZE = 100
# define the raw count percentile threshold for a word to be included in the normalized count dictionary
#  (larger value will result in higher possibility to include rare words)
RAWCOUNT_FILTER_PERCENTAGE = 0.05

def ReadFastaFolder(path):
    def _readfastafile(fasta_file):
        with open(fasta_file, 'r') as f:
            seq = f.readlines()[-1].strip()
        return seq
    file_list = [i for i in os.listdir(path) if i[-6:]=='.fasta']
    return {i[:-6]: _readfastafile(os.path.join(path, i)) for i in file_list}


def LouvainTimeLimit(adj_matrix):
    @TimeOut(5)
    def limited_louvain(graph, seed=42):
        return nx.algorithms.community.louvain_communities(graph, seed=seed)
    G = nx.DiGraph()
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    seed = 1
    while seed <= 3:
        try:
            partition = limited_louvain(G, seed=seed)
            return partition
        except TimeoutError:
            seed += 1
        except ZeroDivisionError:
            break
    partition = []
    # partition, _ = louvain_method(adj_matrix)
    return partition


def DensityFilter(matrix, comm, threshold=0.01):
    density_cal = matrix[list(comm),:][:, list(comm)]
    density_cal[np.arange(len(comm)), np.arange(len(comm))]=0
    return np.mean(density_cal) > threshold


def ConnectivityFilter(matrix, comm):
    connectivity = connected_components(matrix[list(comm), :][:, list(comm)])[0]==1
    return connectivity


def SortByPageRank(graph_matrix):
    num_nodes = graph_matrix.shape[0]
    G = nx.DiGraph()
    rows, cols = np.where(graph_matrix==1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    pagerank_dict = nx.pagerank(G)
    for i in range(num_nodes):
        if i not in pagerank_dict.keys():
            pagerank_dict[i] = 0
    pagerank_values = [pagerank_dict[i] for i in range(num_nodes)]
    return np.argsort(pagerank_values)[::-1]


def UselessHeadParse(head_list):
    heads_disc = np.zeros((33,20))
    useless_heads = [[int(i) for i in head[1:].split('H')] for head in head_list]
    for i,j in useless_heads:
        heads_disc[i,j] = 1
    return heads_disc


def single_seq_community_detection(attentions, chain_ignored_heads):
    branches_louvain = np.zeros((attentions.shape[0], attentions.shape[1]), dtype=object)
    for layer in range(attentions.shape[0]):
        for head in range(attentions.shape[1]):
            if chain_ignored_heads[layer][head]:
                continue
            comm = LouvainTimeLimit(attentions[layer][head])
            comm = [i for i in comm if len(i) > 1]
            comm = [i for i in comm if DensityFilter(attentions[layer][head], i)]
            comm = [i for i in comm if ConnectivityFilter(attentions[layer][head], i)]
            branches_louvain[layer][head] = comm
    return branches_louvain


def single_seq_wordlist(chain_name, sequence, attentions, branches_louvain, chain_ignored_heads):
    words_full_list = []
    sequence_array = np.array([i for i in sequence])
    sequence_simplified = np.array([i for i in ResidueClassification(sequence, 3)])
    for layer in range(attentions.shape[0]):
        for head in range(attentions.shape[1]):
            if chain_ignored_heads[layer][head]:
                continue
            for branch in branches_louvain[layer][head]:
                node_num = len(branch)
                if node_num >= 5 and node_num <= 20:
                    np_branch = np.array(list(branch))
                    np_branch_sorted = np.sort(np_branch)

                    topo_branch = attentions[layer, head][np_branch_sorted][:, np_branch_sorted]
                    word_connections = np.sum(topo_branch)
                    word_pagerank = SortByPageRank(topo_branch)

                    word_seq_rescl = sequence_simplified[np_branch_sorted]
                    word_seq = sequence_array[np_branch_sorted]
                    word_type = ''.join(np.sort([i for i in word_seq_rescl]))
                    word_type_sorted_by_pos = ''.join(np.array([i for i in word_seq_rescl]))
                    word_seq_sorted_by_pos = ''.join(np.array([i for i in word_seq]))
                    word_type_sorted_pagerank = ''.join(np.array([i for i in word_seq_rescl])[word_pagerank])
                    word_seq_sorted_pagerank = ''.join(np.array([i for i in word_seq])[word_pagerank])
                    words_full_list.append([chain_name, layer, head, word_type, \
                                            np_branch_sorted, word_type_sorted_by_pos, word_seq_sorted_by_pos, \
                                            word_connections, word_pagerank,
                                            word_type_sorted_pagerank, word_seq_sorted_pagerank])
    word_dataframe = pd.DataFrame(words_full_list)
    word_dataframe.columns = ['chain', 'layer', 'head', 'word_type', 'pos', 'seq_restype', 'seq', 'num_edges',
                              'pagerank', 'seq_restype_by_pagerank', 'seq_by_pagerank']
    return word_dataframe


def dictionary_raw_counts(word_dataframe, dict_size=100):
    sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    dictionary = sequence_compositions[0][np.argsort(sequence_compositions[1])[::-1][:dict_size]]

    # Save normalized dictionary and sequence segmentation table
    word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(dictionary)]
    return dictionary, word_dataframe_bycount


def restype_freq_baseline(seq_dict_all):
    # Calculate frequency of each degenerated residue type
    all_types = np.array([]).astype(str)
    for sequence in seq_dict_all.values():
        sequence_simplified = np.array([i for i in ResidueClassification(sequence, 3)])
        all_types = np.concatenate((all_types, sequence_simplified))
    res_prob_baseline = np.unique(all_types, return_counts=True)[1] / np.sum(
        np.unique(all_types, return_counts=True)[1])
    res_prob_baseline = dict(zip(np.unique(all_types, return_counts=True)[0], res_prob_baseline))
    return res_prob_baseline


def dictionary_normalized_counts(word_dataframe, res_prob_baseline, rawcount_filter=0.05, dict_size=100):
    sequence_compositions = np.unique(np.array(word_dataframe['word_type']), return_counts=True)
    # Calculate normalized count for every unique words
    wordfreq_compensated = []
    for i, word in enumerate(sequence_compositions[0]):
        word_counts = sequence_compositions[1][i]
        wordfreq_compensated.append(word_counts / np.prod([res_prob_baseline[i] * len(res_prob_baseline) for i in word]))

    # Create a DataFrame of all words' raw and normalized count
    wordfreq_compensated = np.array(wordfreq_compensated)
    dictionary_dataframe = pd.DataFrame([sequence_compositions[0], sequence_compositions[1], wordfreq_compensated]).T
    dictionary_dataframe.columns = ['sequence', 'raw_count', 'normalized']
    dictionary_dataframe = dictionary_dataframe.sort_values(by=['raw_count', 'normalized'])
    dictionary_dataframe = dictionary_dataframe.iloc[-int(dictionary_dataframe.shape[0] * rawcount_filter):]
    common_compositions_corrected = list(dictionary_dataframe.sort_values(by='normalized').iloc[-dict_size:]['sequence'])
    word_dataframe_bycount = word_dataframe.loc[word_dataframe['word_type'].isin(common_compositions_corrected)]
    return common_compositions_corrected, word_dataframe_bycount

def extract_words(seq_dict_all, basename, attns_dict, output_path, ignored_heads):
    # run script command format: python community_to_dictionary.py <fasta_path> <embedding_path> <output_filename>
    '''file format:
    sequence: <fasta_path>/<chain1>.fasta, <chain2>.fasta ...
    attention: <embedding_path>/<chain1>_all.tar.gz , <chain2>_all.tar.gz ...
    '''
    # Step 0: Read files and determine chain names
    # attention_map_file_names = [i for i in os.listdir(embedding_path) if i.endswith('all_heads.pkl')]
    # chain_names = [i.split('_all_heads')[0] for i in attention_map_file_names if i.split('_all_heads')[0] in seq_dict_all.keys()]
    chain_names = list(seq_dict_all.keys())
    words_full_list = []

    # Step 1: Community discovery
    for cnum, chain_name in enumerate(chain_names):
        print('Start to extract words :', chain_name)
        sequence = seq_dict_all[chain_name]
        attentions = attns_dict[chain_name]  # pd.read_pickle(embedding_path+'/'+chain_name+'_all_heads.pkl')
        chain_ignored_heads = ignored_heads[chain_name]
        #   Louvain: branches_louvain
        branches_louvain = single_seq_community_detection(attentions, chain_ignored_heads)

        word_dataframe_single = single_seq_wordlist(chain_name, sequence, attentions, branches_louvain, ignored_heads)
        words_full_list.append(word_dataframe_single)

    # Step 2: Create table of communities and their metadata
    word_dataframe = pd.concat(words_full_list)
    word_dataframe.to_csv(os.path.join(output_path, f'{basename}_dict_raw.csv'))

    # Step 3: Create word dictionary and segment table
    dictionary, word_dataframe_bycount = dictionary_raw_counts(word_dataframe)

    # Save normalized dictionary and sequence segmentation table
    np.save(os.path.join(output_path, f'{basename}_dictionary_rawcount.npy'), dictionary)
    word_dataframe_bycount.to_csv(os.path.join(output_path, f'{basename}_segment_table_rawcount.csv'))


    # Step 4: Create word dictionary using normalized count
    res_prob_baseline = restype_freq_baseline(seq_dict_all)
    dictionary_normalized, word_dataframe_normalized = dictionary_normalized_counts(word_dataframe, res_prob_baseline)

    # Save normalized dictionary and sequence segmentation table
    np.save(os.path.join(output_path, f'{basename}_dictionary_normalized.npy'), dictionary_normalized)
    word_dataframe_normalized.to_csv(os.path.join(output_path, f'{basename}_segment_table_normalized.csv'))

    return word_dataframe_bycount

