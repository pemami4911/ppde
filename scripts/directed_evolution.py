import argparse
import torch
torch.set_printoptions(threshold=5000)
import random
import numpy as np
import os
import datetime
import json 
from pathlib import Path 

from ppde.energy import ProteinProductOfExperts, ProteinSupervised
from ppde.nets import AugmentedLinearRegression
from ppde.protein_samplers.ppde import PPDE_PAS
from ppde.protein_samplers.sa import SimulatedAnnealing
from ppde.protein_samplers.mala_approx import MALAApprox
from ppde.protein_samplers.cmaes import CMAES
from ppde.protein_samplers.random import RandomSampler
from ppde.third_party.hsu import io_utils, data_utils
from ppde.metrics import proteins_potts_score, proteins_transformer_score

def get_sampler(args):
    if args.sampler == 'simulated_annealing':
        return SimulatedAnnealing(args)
    elif args.sampler == 'PPDE':
        return PPDE_PAS(args)
    elif args.sampler == 'MALA-approx':
        return MALAApprox(args)
    elif args.sampler == 'CMAES':
        return CMAES(args)
    elif args.sampler == 'Random':
        return RandomSampler(args)


def main(args):
    """
    Run sampler
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.hub.set_dir(args.hub_dir)

    if args.run_signature == '':
        unique_token = "{}_{}_{}".format(
            args.sampler, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        unique_token = "{}_{}_{}_{}".format(
            args.sampler, args.run_signature, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    results_path = Path(args.results_path, args.protein, unique_token)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load energy function models
    if args.energy_function == 'product_of_experts':
        energy_func = ProteinProductOfExperts(args)
    elif args.energy_function == 'supervised':
        energy_func = ProteinSupervised(args)

    energy_func = energy_func.to(args.device)

    # Load oracle
    oracle = AugmentedLinearRegression(
        os.path.join(args.protein_weights, args.protein))
    oracle.to(args.device)
    
    ###### create the initial population #########
    wtseqs = io_utils.read_fasta(os.path.join(args.protein_weights, args.protein, 'wt.fasta'), return_ids=False)   
    initial_population = torch.from_numpy(data_utils.seqs_to_onehot(wtseqs)).float().to(args.device)
    initial_population = initial_population.repeat(args.n_chains,1,1)

    with torch.no_grad():
        print(f'WT protein energy: {energy_func.get_energy(initial_population)[0].mean():.3f}')

    sampler = get_sampler(args)
    
    ###### run sampler #########        
    best_samples, best_energy, best_fitness, energy_history, fitness_history, random_traj = \
        sampler.run(initial_population, args.n_iters, energy_func, oracle.potts.index_list[0],
                    oracle.potts.index_list[-1], oracle, args.log_every)
    
    best_oracle = oracle(best_samples).detach().cpu().numpy()
    potts_score = proteins_potts_score(best_samples, os.path.join(args.protein_weights, args.protein)).cpu().numpy()
    
    print(f'energy quantiles: {np.quantile(best_energy, [0.2,0.4,0.6,0.8,1.0])}')
    print(f'fitness quantiles: {np.quantile(best_fitness, [0.2,0.4,0.6,0.8,1.0])}')
    print(f'oracle quantiles: {np.quantile(best_oracle, [0.2,0.4,0.6,0.8,1.0])}')
    print(f'potts quantiles: {np.quantile(potts_score, [0.2,0.4,0.6,0.8,1.0])}')

    # dump config to file
    with open(results_path / 'config.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    np.save(results_path / 'population.npy', best_samples.detach().cpu().numpy())
    np.save(results_path / 'pred_fitness_scores.npy', best_fitness)
    np.save(results_path / 'oracle_fitness_scores.npy', best_oracle)
    np.save(results_path / 'potts_scores.npy', potts_score)
    np.save(results_path / 'energy_scores.npy', best_energy)
    np.save(results_path / 'energy_history.npy', energy_history)
    np.save(results_path / 'fitness_history.npy', fitness_history)
    
    if not args.disable_MSA_transformer_scoring:
        transformer_score = proteins_transformer_score(best_samples, os.path.join(args.protein_weights, args.protein), 
                                args.msa_path, args.msa_size)
        print(f'MSATransformer quantiles: {np.quantile(transformer_score, [0.2,0.4,0.6,0.8,1.0])}')
        np.save(results_path / 'transformer_scores.npy', transformer_score)

    print('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # general
    general_args = parser.add_argument_group('general')
    general_args.add_argument('--protein_weights', type=str, default='weights')
    general_args.add_argument('--results_path', type=str, default='results/proteins')
    general_args.add_argument('--protein', type=str, default='PABP_YEAST_Fields2013',
                              help='PABP_YEAST_Fields2013,'
                              'UBE4B_MOUSE_Klevit2013-nscor_log2_ratio,'
                              'GFP_AEQVI_Sarkisyan2016')
    general_args.add_argument('--hub_dir', type=str, default='.')
    general_args.add_argument('--msa_path', type=str, default='data/proteins/PABP_YEAST.a2m')
    general_args.add_argument('--msa_size', type=int, default=500)
    general_args.add_argument('--seed', type=int, default=1234567)
    general_args.add_argument('--device', type=str, default='cuda')
    general_args.add_argument('--log_every', type=int, default=50)
    general_args.add_argument('--run_signature', type=str, default='')

    general_args.add_argument('--n_iters', type=int, default=10000)
    general_args.add_argument('--n_chains', type=int, default=128)
    general_args.add_argument('--energy_lamda', type=float, default=5)
    general_args.add_argument('--energy_function', type=str, default='product_of_experts',
                                help='product_of_experts, supervised')
    general_args.add_argument('--unsupervised_expert', type=str, default='potts',
                                help='potts, transformer-S, transformer-M, transformer-L, '
                                ' potts+transformer')    
    general_args.add_argument('--sampler', type=str, default='PPDE')    
    general_args.add_argument('--nmut_threshold', type=int, default=0,
                               help='Enforce a maximum number of mutations to WT'
                               'disabled by setting to 0')
    general_args.add_argument('--disable_MSA_transformer_scoring', action='store_true',
                               help='Disable MSATransformer scoring. Useful for debugging '
                               'or if you don\'t have a GPU. Weights are a ~1.3 GB download.')
    general_args.add_argument('--paper_results', action='store_true', default=False,
                               help='Reproduce paper results by resetting Markov chain instead of rejecting proposal')
    
    # sampler (simulated annealing)
    sa_args = parser.add_argument_group('simulated_annealing')
    sa_args.add_argument('--simulated_annealing_temp', type=float, default=0.01)
    sa_args.add_argument('--muts_per_seq_param', type=float, default=1.5)
    sa_args.add_argument('--decay_rate', type=float, default=0.999)

    diffusion_args = parser.add_argument_group('mala_approx')
    diffusion_args.add_argument('--diffusion_step_size', type=float, default=0.1)
    diffusion_args.add_argument('--diffusion_relaxation_tau', type=float, default=0.99)

    cmaes_args = parser.add_argument_group('cmaes')
    cmaes_args.add_argument('--cmaes_population_size', type=int, default=16)
    cmaes_args.add_argument('--cmaes_initial_variance', type=float, default=0.05)

    pppo_args = parser.add_argument_group('ppde')
    pppo_args.add_argument('--ppde_pas_length', type=int, default=2)

    args = parser.parse_args()

    main(args)
