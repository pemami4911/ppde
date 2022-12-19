import argparse
import torch
torch.set_printoptions(threshold=5000)
import random
import numpy as np
import os
import datetime
import json 

from src.energy import ProteinJointEnergy, ProteinEnergy
from src.nets import AugmentedLinearRegression
from src.protein_samplers.ppde import PPDE_PAS
from src.protein_samplers.sa import SimulatedAnnealing
from src.protein_samplers.mala_approx import MALAApprox
from src.protein_samplers.cmaes import CMAES
from src.protein_samplers.random import RandomSampler
from src.third_party.hsu import io_utils, data_utils
from src.metrics import proteins_potts_score, proteins_transformer_score

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

    # Load energy function models
    if args.energy_function == 'joint':
        energy_func = ProteinJointEnergy(args)
    elif args.energy_function == 'fitness':
        energy_func = ProteinEnergy(args)

    energy_func = energy_func.to(args.device)

    # Load oracle
    oracle = AugmentedLinearRegression(os.path.join(args.data_dir, 'weights', args.dataset_name))
    oracle.to(args.device)
    
    ###### create the initial population #########
    wtseqs = io_utils.read_fasta(os.path.join(args.data_dir, 'weights', args.dataset_name, 'wt.fasta'), return_ids=False)   
    initial_population = torch.from_numpy(data_utils.seqs_to_onehot(wtseqs)).float().to(args.device)
    initial_population = initial_population.repeat(args.n_chains,1,1)

    with torch.no_grad():
        print(f'WT protein energy: {energy_func.get_energy(initial_population)[0].mean():.3f}')

    sampler = get_sampler(args)
    
    if args.run_signature == '':
        unique_token = "{}_{}_{}".format(
            args.sampler, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        unique_token = "{}_{}_{}_{}".format(
            args.sampler, args.run_signature, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    results_dir = os.path.join(args.results_path, unique_token)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #abbrv += f'_{args.energy_function}'
    #if args.suffix != '':
    #    abbrv += f'_{args.suffix}'

    ###### run sampler #########        
    best_samples, best_energy, best_fitness, energy_history, fitness_history, random_traj = \
        sampler.run(initial_population, args.n_iters, energy_func, oracle.potts.index_list[0],
                    oracle.potts.index_list[-1], oracle, None, args.log_every)
    
    best_oracle = oracle(best_samples).detach().cpu().numpy()
    potts_score = proteins_potts_score(best_samples, os.path.join(args.data_dir, 'weights', args.dataset_name)).cpu().numpy()
    transformer_score = proteins_transformer_score(best_samples, os.path.join(args.data_dir, 'weights', args.dataset_name), 
                            os.path.join(args.data_dir, args.msa_path), args.msa_size)
    print(f'energy quantiles: {np.quantile(best_energy, [0.2,0.4,0.6,0.8,1.0])}')
    print(f'fitness quantiles: {np.quantile(best_fitness, [0.2,0.4,0.6,0.8,1.0])}')
    print(f'oracle quantiles: {np.quantile(best_oracle, [0.2,0.4,0.6,0.8,1.0])}')
    print(f'potts quantiles: {np.quantile(potts_score, [0.2,0.4,0.6,0.8,1.0])}')
    print(f'MSATransformer quantiles: {np.quantile(transformer_score, [0.2,0.4,0.6,0.8,1.0])}')

    # write to file
    # 0. args
    # 1. population as one-hot numpy array
    # 2. best oracle scores
    # 3. potts scores 
    # 4. best predicted fitness scores
    # 5. best energy scores
    with open(os.path.join(results_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    np.save(os.path.join(results_dir, 'population.npy'), best_samples.detach().cpu().numpy())
    np.save(os.path.join(results_dir, 'pred_fitness_scores.npy'), best_fitness)
    np.save(os.path.join(results_dir, 'oracle_fitness_scores.npy'), best_oracle)
    np.save(os.path.join(results_dir, 'potts_scores.npy'), potts_score)
    np.save(os.path.join(results_dir, 'energy_scores.npy'), best_energy)
    np.save(os.path.join(results_dir, 'transformer_scores.npy'), transformer_score)
    np.save(os.path.join(results_dir, 'energy_history.npy'), energy_history)
    np.save(os.path.join(results_dir, 'fitness_history.npy'), fitness_history)
    
    print('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # general
    general_args = parser.add_argument_group('general')
    general_args.add_argument('--data_dir', type=str, default='/gpfs/alpine/bie108/proj-shared/pppo')
    general_args.add_argument('--results_path', type=str, default='./experiments/proteins/results')
    general_args.add_argument('--dataset_name', type=str, default='PABP_YEAST_Fields2013',
                              help='PABP_YEAST_Fields2013,UBE4B_MOUSE_Klevit2013-nscor_log2_ratio,GFP_AEQVI_Sarkisyan2016')
    general_args.add_argument('--hub_dir', type=str, default='/gpfs/alpine/bie108/proj-shared/torch/hub/')
    general_args.add_argument('--msa_path', type=str, default='alignments/PABP_YEAST.a2m', help='args.data_dir + args.msa_path')
    general_args.add_argument('--msa_size', type=int, default=500)
    general_args.add_argument('--seed', type=int, default=1234567)
    general_args.add_argument('--device', type=str, default='cuda')
    general_args.add_argument('--log_every', type=int, default=50)
    general_args.add_argument('--run_signature', type=str, default='')

    general_args.add_argument('--n_iters', type=int, default=10000)
    general_args.add_argument('--n_chains', type=int, default=128)
    general_args.add_argument('--prior', type=str, default='potts')    
    general_args.add_argument('--energy_lamda', type=float, default=5)
    general_args.add_argument('--energy_function', type=str, default='joint')
    general_args.add_argument('--sampler', type=str, default='PPDE')    
    general_args.add_argument('--nmut_threshold', type=int, default=2)
    
    # sampler (simulated annealing)
    sa_args = parser.add_argument_group('simulated_annealing')
    sa_args.add_argument('--simulated_annealing_temp', type=float, default=0.01)
    sa_args.add_argument('--muts_per_seq_param', type=float, default=1.5)
    sa_args.add_argument('--decay_rate', type=float, default=0.999)

    diffusion_args = parser.add_argument_group('MALA')
    diffusion_args.add_argument('--diffusion_step_size', type=float, default=0.1)
    diffusion_args.add_argument('--diffusion_relaxation_tau', type=float, default=0.99)

    pppo_args = parser.add_argument_group('ppde')
    pppo_args.add_argument('--ppde_gwg_samples', type=int, default=1)
    pppo_args.add_argument('--ppde_pas_length', type=int, default=1)

    cmaes_args = parser.add_argument_group('cmaes')
    cmaes_args.add_argument('--cmaes_population_size', type=int, default=16)
    cmaes_args.add_argument('--cmaes_initial_variance', type=float, default=0.05)

    args = parser.parse_args()

    main(args)
