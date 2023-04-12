import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ppde.nets import PottsModel
from ppde.utils import load_MSA
from ppde.third_party.hsu import data_utils
from esm_one_hot import pretrained
from tqdm import trange


def proteins_potts_score(population, dataset_name):
    """ delta hamiltonian """
    potts = PottsModel(dataset_name).to(population.device)
    potts.eval()
    with torch.no_grad():
        return potts(potts.preprocess_onehot(population), delta=True)


def proteins_transformer_score(population, dataset_name, msa_location, msa_size):
    potts = PottsModel(dataset_name).to(population.device)
    align_idx = potts.index_list
    wt = potts.wtseqs[0]   
    print('loading the MSA...')
    msa = load_MSA(msa_location)
    # subsample msa to msa_size
    idxs = np.random.choice(list(range(len(msa))), size=msa_size-1, replace=False)
    msa = [msa[i] for i in idxs]
    print('loading the MSATransformer...')
    esm, esm_alphabet = pretrained.esm_msa1b_t12_100M_UR50S()
    esm=esm.to(population.device)
    esm.eval()
    batch_converter = esm_alphabet.get_batch_converter()

    population = data_utils.onehot2seq(population.cpu().numpy())

    
    ################ Masked marginal scoring #################
    # sum over # mutations of the log likelihood ratio between mutant and wt
    # we mask the location of the mutation. assumes mutation effects are additive.
    scores = []
    print('scoring population with the MSATransformer...')
    for i in trange(len(population)):

        # for each mutant
        muts = data_utils.seq2mutation_fromwt(population[i], wt)
        #print(f'{i}, muts = {muts}')
        mut_sum=0
        for l in range(len(muts)):
            mask_idx = muts[l][0]
            wt_aa = muts[l][1]
            mut_aa = muts[l][2]
            # skipping mutations outside of alignment window
            if mask_idx < align_idx[0] or mask_idx > align_idx[-1]:
                continue
            # mask out mut aa for first sequence, first with *
            masked_seq = wt[:mask_idx] + '*' + wt[mask_idx+1:]
            # select mutant subsequence aligned to the MSA
            masked_seq = masked_seq[align_idx[0]:align_idx[-1]+1]
            # replace * with <mask>
            masked_seq=masked_seq.replace('*','<mask>')
            msa_ = [('mut',masked_seq)] + msa
            labels, strs, one_hot_msa = batch_converter(msa_, align_idx[-1]-align_idx[0]+1)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    token_probs = torch.log_softmax(
                        esm(one_hot_msa.cuda())["logits"], dim=-1
                    )                 
                    row = 1 + mask_idx - align_idx[0]   
                    mut_sum += (token_probs[0,0,row, esm_alphabet.get_idx(mut_aa)] \
                                - token_probs[0,0,row, esm_alphabet.get_idx(wt_aa)]).item()
        scores += [mut_sum]
        print(mut_sum)
    return np.array(scores)

def n_hops(population, wt):
    K, L, V = population.shape
    nhops = []
    for k in range(K):
        prot = population[k]
        diff = ((prot - wt) > 0).float()
        nhops += [diff.sum()]
    return torch.mean(torch.stack(nhops)), torch.std(torch.stack(nhops))

def score_diversity(population, autoencoder):
    """Compute average Euclidean distance between i and all pairs j
    
    population is [K, 1, 28, 28]
    """
    with torch.no_grad():
        # encode --> [K,D]
        K = population.size(0)
        population = population.view(K,-1)
        embeddings = autoencoder.encode(population).mean
        D = embeddings.size(1)
        distance = torch.norm( embeddings.view(K,1,D) - embeddings.view(1,K,D), 2, 2)  # [K,K]
        avg_distance = torch.sum(distance) / (K**2 - K)
        return avg_distance
    

def mnist_scores_to_csv(pred_scores, oracle_scores, method, args):
    pred_score_quantiles = np.quantile(pred_scores, [0.5, 0.6, 0.7, 0.8, 0.9], axis=1)  # [num_steps, 5]
    gt_score_quantiles = np.quantile(oracle_scores, [0.5, 0.6, 0.7, 0.8, 0.9], axis=1)  # [num_steps, 5]
    xs = [i*args.log_every for i in range(pred_scores.shape[0])]
    xs = np.array(xs)
    csv_file = args.results_path + '/' + method + f'_pred_sums.csv'
    df = pd.DataFrame(pred_score_quantiles.T, columns = ['0.5', '0.6', '0.7', '0.8', '0.9'], index=xs)
    df.to_csv(csv_file)
    csv_file = args.results_path + '/' + method + f'_oracle_sums.csv'
    df = pd.DataFrame(gt_score_quantiles.T, columns = ['0.5', '0.6', '0.7', '0.8', '0.9'], index=xs)
    df.to_csv(csv_file)

def mnist_performance_plots(pred_scores, oracle_scores, method, args):
    pred_score_quantiles = np.quantile(pred_scores, [0.5, 0.6, 0.7, 0.8, 0.9], axis=1)
    gt_score_quantiles = np.quantile(oracle_scores, [0.5, 0.6, 0.7, 0.8, 0.9], axis=1)
    xs = [i*args.log_every for i in range(pred_scores.shape[0])]
    #plt.plot(xs, pred_score_quantiles[1], label=f'{method} @ 50%', linestyle='--')
    #plt.plot(xs, gt_score_quantiles[1], label=f'(GT) {method} @ 50%')
    #plt.plot(xs, pred_score_quantiles[3], label=f'{method} @ 90%', linestyle='--')
    #plt.plot(xs, gt_score_quantiles[3], label=f'(GT) {method} @ 90%')
    plt.plot(xs, pred_score_quantiles[2], label=f'pred.', linestyle='--')
    plt.fill_between(xs, pred_score_quantiles[0], pred_score_quantiles[-1], alpha=.1, linewidth=1)
    plt.plot(xs, gt_score_quantiles[2], label=f'oracle')
    plt.fill_between(xs, gt_score_quantiles[0], gt_score_quantiles[-1], alpha=.1, linewidth=1)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
    #plt.ylim(9,21)
    plt.xlabel('step')
    plt.ylabel('sum')
    plt.tight_layout()
    plt.savefig(args.results_path + '/' + method + '_scores.pdf')
    plt.savefig(args.results_path + '/' + method + '_scores.png')