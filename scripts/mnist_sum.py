import torch
import torchvision
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import random 
from pathlib import Path 

from ppde.energy import MNISTProductOfExperts, MNISTSupervised
from ppde.mnist_samplers.sa import SimulatedAnnealing
from ppde.nets import MNISTRegressionNet
from ppde.mnist_samplers.mala_approx import MALAApprox
from ppde.mnist_samplers.cmaes import CMAES
from ppde.mnist_samplers.ppde import PPDE
from ppde.metrics import mnist_scores_to_csv, mnist_performance_plots


def get_sampler(args):
    if args.sampler == 'simulated_annealing':
        return SimulatedAnnealing(args), "SA"
    elif args.sampler == 'PPDE':
        if args.ppde_pas_length > 0:
            abbrv = f'PPDE-PAS-{args.ppde_pas_length}'
        else:
            abbrv = f'PPDE-GWG-{args.ppde_gwg_samples}'
        return PPDE(args), abbrv
    elif args.sampler == 'MALA-approx':
        return MALAApprox(args), "MALA-approx"
    elif args.sampler == 'CMAES':
        return CMAES(args), 'CMAES'
        

def make_gif(xs, method, args):
    """
    Take a sequence of MNIST images and convert to a movie
    """
    xs = [Image.fromarray((255 * x[:,:,0]).astype(np.uint8)).convert('P') for x in xs]
    xs[0].save(
        args.results_path + '/' + method + '.gif',
        save_all=True,
        append_images=xs[1:],
        duration=100, loop=0)

def visualize_population(population, method, args):
    grid = torchvision.utils.make_grid(population.view(-1,28,28).unsqueeze(1))
    plt.figure(figsize=(6,10))
    plt.imshow(grid.permute(1,2,0).numpy())

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(args.results_path + '/' + method + '_final_population.pdf')
    plt.savefig(args.results_path + '/' + method + '_final_population.png')
    
    np.save(args.results_path + '/' + method + '_final_population.npy', population.view(-1,28,28).numpy())


def main(args):
    """
    Run sampler
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # Create results directory
    Path(args.results_path).mkdir(parents=True, exist_ok=True)
    args.mnist_weights = Path(args.mnist_weights)

    # Load energy function models
    if args.energy_function == 'product_of_experts':
        init_mean = torch.from_numpy(np.load('./data/mnist/mnist_mean.npy')).float()
        energy_func = MNISTProductOfExperts(init_mean, args)
    elif args.energy_function == 'supervised':
        energy_func = MNISTSupervised(args)

    energy_func = energy_func.to(args.device)

    # Load oracle
    oracle = MNISTRegressionNet(64)
    oracle.load_state_dict(torch.load(
        args.mnist_weights / 'one-hot_GT_ckpt_60000.pt',
        map_location=torch.device(args.device))['model'])
    oracle.to(args.device)

    ###### create the initial population #########
    if args.wild_type == 0:
        summand_a = torch.from_numpy(np.load('./data/mnist/3_0.npy')).float()
        summand_b = torch.from_numpy(np.load('./data/mnist/3_1.npy')).float()
    elif args.wild_type == 1:
        summand_a = torch.from_numpy(np.load('./data/mnist/29_0.npy')).float()
        summand_b = torch.from_numpy(np.load('./data/mnist/29_1.npy')).float()
    elif args.wild_type == 2:
        summand_a = torch.from_numpy(np.load('./data/mnist/38_0.npy')).float()
        summand_b = torch.from_numpy(np.load('./data/mnist/38_1.npy')).float()
    elif args.wild_type == 3:
        summand_a = torch.from_numpy(np.load('./data/mnist/99_0.npy')).float()
        summand_b = torch.from_numpy(np.load('./data/mnist/99_1.npy')).float()
    elif args.wild_type == 4:
        summand_a = torch.from_numpy(np.load('./data/mnist/149_0.npy')).float()
        summand_b = torch.from_numpy(np.load('./data/mnist/149_1.npy')).float()
    elif args.wild_type == -1:
        summand_a = torch.from_numpy(np.load('./data/mnist/validation_0.npy')).float()
        summand_b = torch.from_numpy(np.load('./data/mnist/validation_1.npy')).float()

    summand_a = summand_a.view(784)
    summand_b = summand_b.view(784)

    #grid = torchvision.utils.make_grid(torch.cat([summand_a.view(-1,1,28,28), summand_b.view(-1,1,28,28)], 2))
    #plt.imshow(grid.permute(1,2,0))

    summand_a = summand_a.unsqueeze(0).repeat(args.n_chains,1)
    summand_b = summand_b.unsqueeze(0).repeat(args.n_chains,1)  # [N,28*28]
    init_seqs = torch.cat((summand_a, summand_b), 1).to(args.device)  # [N,2*(784)]

    sampler, abbrv = get_sampler(args)
    abbrv += f'_{args.energy_function}'
    if args.suffix != '':
        abbrv += f'_{args.suffix}'
    
    final_pop, energy_history, sum_history, oracle_history, random_traj = \
        sampler.run(init_seqs, args.n_iters, energy_func, 0, 784, oracle, args.log_every)
    
    metrics = args.metrics.split('+')
    if 'plots' in metrics:
        mnist_performance_plots( sum_history, oracle_history, abbrv, args)
    if 'viz' in metrics:
        visualize_population( torch.from_numpy(final_pop), abbrv, args )
    if 'csv' in metrics:
        mnist_scores_to_csv( sum_history, oracle_history, abbrv, args)
    if 'gif' in metrics:
        make_gif( random_traj, abbrv, args )
        
    print('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # general
    general_args = parser.add_argument_group('general')
    general_args.add_argument('--mnist_weights', type=str, default='weights/mnist_models')
    general_args.add_argument('--results_path', type=str, default='results/mnist')
    general_args.add_argument('--wild_type', type=int, default=0, help='which mnist pair to use [0, 4]')
    general_args.add_argument('--seed', type=int, default=1234567)
    general_args.add_argument('--device', type=str, default='cuda')
    general_args.add_argument('--n_iters', type=int, default=200)
    general_args.add_argument('--n_chains', type=int, default=128)
    general_args.add_argument('--energy_lamda', type=float, default=10)
    general_args.add_argument('--energy_function', type=str, default='product_of_experts')
    general_args.add_argument('--unsupervised_expert', type=str, default='ebm', help='ebm or dae')
    general_args.add_argument('--log_every', type=int, default=50)
    general_args.add_argument('--sampler', type=str, default='simulated_annealing')
    general_args.add_argument('--suffix', type=str, default='')
    general_args.add_argument('--metrics', type=str, default='gif+plots+viz+csv', 
                              help='which metrics to compute (gif, plots, viz, csv)')

    # sampler (simulated annealing)
    sa_args = parser.add_argument_group('simulated_annealing')
    sa_args.add_argument('--simulated_annealing_temp', type=float, default=10)
    sa_args.add_argument('--muts_per_seq_param', type=float, default=5)
    sa_args.add_argument('--decay_rate', type=float, default=0.999)

    diffusion_args = parser.add_argument_group('mala_approx')
    diffusion_args.add_argument('--diffusion_step_size', type=float, default=0.01)
    diffusion_args.add_argument('--diffusion_relaxation_tau', type=float, default=0.9)

    cmaes_args = parser.add_argument_group('cmaes')
    cmaes_args.add_argument('--cmaes_population_size', type=int, default=16)
    cmaes_args.add_argument('--cmaes_initial_variance', type=float, default=0.1)

    pppo_args = parser.add_argument_group('pppo')
    pppo_args.add_argument('--ppde_gwg_samples', type=int, default=1)
    pppo_args.add_argument('--ppde_pas_length', type=int, default=10)

    args = parser.parse_args()

    main(args)
