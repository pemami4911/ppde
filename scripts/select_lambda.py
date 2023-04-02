# Randomly select 100 proteins < WT and 100 > WT.
# Score with predictor and prior
# Select lambda out of (0.1,1,10,100?) to make (min,max) nearly same
import argparse
import os
import pandas as pd
from ppde.energy import ProteinJointEnergy, ProteinEnergy
import ppde.third_party.hsu.data_utils as data_utils
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda')    
    parser.add_argument('--data_dir', type=str, default='/gpfs/alpine/bie108/proj-shared/pppo')
    parser.add_argument('--results_path', type=str, default='./experiments/proteins/results')
    parser.add_argument('--dataset_name', type=str, default='PABP_YEAST_Fields2013')
    parser.add_argument('--prior', type=str, default='potts')    
    parser.add_argument('--energy_lamda', type=float, default=1)
    parser.add_argument('--n_chains', type=int, default=128)

    args = parser.parse_args()
    # Load energy function models

    energy_func = ProteinJointEnergy(args)
    energy_func = energy_func.to(args.device)

    if args.dataset_name == 'PABP_YEAST_Fields2013':
        dn = 'PABP_YEAST_Fields2013-linear'
    else:
        dn = args.dataset_name
    data_path = os.path.join(args.data_dir, 'processed_data', dn, 'data.csv')
    # Sample shuffles the DataFrame.
    data = pd.read_csv(data_path)
    is_valid = data['seq'].apply(data_utils.is_valid_seq)
    data = data[is_valid]
    X = data.seq
    Y = data.log_fitness

    good = X[Y > 0].sample(n=100)
    bad = X[Y<0].sample(n=100)


    good_seqs = torch.from_numpy(data_utils.seqs_to_onehot(good.values)).float().to(args.device)
    bad_seqs = torch.from_numpy(data_utils.seqs_to_onehot(bad.values)).float().to(args.device)

    pred_vals = []
    prior_vals = []

    with torch.no_grad():
        pred_vals += [energy_func.get_fitness(good_seqs)]
        prior_vals += [energy_func.get_prior(good_seqs)]
        pred_vals += [energy_func.get_fitness(bad_seqs)]
        prior_vals += [energy_func.get_prior(bad_seqs)]

    pred_vals = torch.cat(pred_vals,0)
    prior_vals = torch.cat(prior_vals,0)

    lamda = args.energy_lamda

    print( torch.min(lamda * pred_vals), torch.max(lamda * pred_vals))
    print( torch.min(prior_vals), torch.max(prior_vals))
    print( torch.mean( lamda * pred_vals), torch.std(lamda * pred_vals))
    print( torch.mean( prior_vals), torch.std( prior_vals))