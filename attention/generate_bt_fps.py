import mask
from fairseq.models.roberta import RobertaModel
import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
# global j
# j=0
# global t
# t=0
from mask import mask_matrix
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_files_smiles='data/input_smiles/uspto_muti_1000.smi'
#data_files_smiles='data/input_smiles/examples/disconnect_USPTO50K.smi'
folder_name_smiles=os.path.join(parent_dir, data_files_smiles)


def load_pretrain_model(model_name_or_path, checkpoint_file, data_name_or_path, bpe='smi'):
    '''Currently only load to cpu()'''

    # load model
    pretrain_model = RobertaModel.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,  # dict_dir,
        bpe=bpe,
    )
    pretrain_model.eval()
    return pretrain_model


def extract_hidden(pretrain_model, target_file, mask_matrix=None):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    hidden_features = {i: None for i in range(sample_num)}

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue

        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        _, all_layer_hiddens,attn, attn_8head = pretrain_model.model(
            tokens.unsqueeze(0), features_only=True, return_all_hiddens=True, mask_matrix=mask_matrix)


        hidden_info = all_layer_hiddens['inner_states'][-1]
        # last_hidden shape [tokens_num, sample_num(default=1), hidden_dim]

        line_dit = line.strip()

        line_dit_addr=line_dit.replace("/", "x")# use q replace '/' for save npy,x is not in dict
        data_last_npy = f'data/middle_attention/npy_supervision/uspto/attention/{line_dit_addr}.npy'
        folder_last_attention = os.path.join(parent_dir, data_last_npy)
        last_attention=hidden_info.cpu().detach().numpy()
        last_attention_dim=np.squeeze(last_attention,axis=1)
        try:
            np.save(folder_last_attention,last_attention_dim)
        except:
            pass

        # hidden_features.append(hidden_info.squeeze(1).cpu().detach().numpy())
        hidden_features[i] = hidden_info.squeeze(1).cpu().detach().numpy()

        #data_files_npy = f'data/middle_attention/npy_property/double_molecule_0911mark/{line_dit}.npy'

        # print(attn_8head)

        np.save(folder_name_attention,attn_8head)
    # hidden_features type: dict, length: samples_num
    return hidden_features ,attn, attn_8head



def extract_features_from_hidden(hidden_info):

    samples_num = len(hidden_info)
    hidden_dim = np.shape(hidden_info[0])[-1]
    samples_features = np.zeros([samples_num, hidden_dim])
    for n_sample, hidden in hidden_info.items():
        # hidden shape [tokens, embed_dim]
        samples_features[n_sample, :] = hidden[0, :]

    return samples_features


def main(args):

    pretrain_model = load_pretrain_model(
        args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.bpe)
    mask_matrix=mask.mask_matrix()
    hidden_info,attn, attn_8head = extract_hidden(pretrain_model, args.target_file, mask_matrix)
    attn_8head = np.array(attn_8head)

    #np.save('result/8_npy/Oc1ccc(OCc2ccccc2)cc1.npy', attn_twodim_array)
    # plt.figure(figsize=(8,8))
    # plt.imshow(attn_twodim,cmap='Greys')
    # plt.suptitle("GridSpec Inside GridSpec")
    # plt.imshow(attn_twodim)
    # plt.show()

    print('Generate features from hidden information')
    samples_features = extract_features_from_hidden(hidden_info)
    print(f'Features shape: {np.shape(samples_features)}')
    np.save(args.save_feature_path, samples_features)
    return(attn_8head)

def parse_args(args):
    parser = argparse.ArgumentParser(description="Tools kit for downstream jobs")

    parser.add_argument('--model_name_or_path', default="./chembl_pubchem_zinc_models/chembl27_512/", type=str,
                        help='Pretrained model folder')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--data_name_or_path', default="./chembl_pubchem_zinc_models/chembl27_512/", type=str,
                        help="Pre-training dataset folder")
    parser.add_argument('--dict_file', default='dict.txt', type=str,
                        help="Pre-training dict filename(full path)")
    parser.add_argument('--bpe', default='smi', type=str)
 #  parser.add_argument('--target_file', default='./examples/data/example_single.smi', type=str,
 #                           help="Target file for feature extraction, default format is .smi")
    parser.add_argument('--target_file', default=folder_name_smiles, type=str,
                        help="Target file for feature extraction, default format is .smi")
    parser.add_argument('--save_feature_path', default='extract_f1.npy', type=str,
                        help="Saving feature filename(path)")
    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    attn_twodim_array=main(args)
    return(attn_twodim_array)


if __name__ == '__main__':
    attn=cli_main()

