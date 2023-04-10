
import cma
import numpy as np
import torch
from ppde.base_sampler import BaseSampler
from ppde.third_party.hsu import data_utils


class CMAES(BaseSampler):
    """
    An explorer which implements the covariance matrix adaptation evolution
    strategy (CMAES).
    Optimizes a continuous relaxation of the one-hot sequence that we use to
    construct a normal distribution around, sample from, and then argmax to get
    sequences for the objective function.
    http://blog.otoro.net/2017/10/29/visual-evolution-strategies/ is a helpful guide.
    """


    def __init__(self, args):
        """
        Args:
            population_size: Number of proposed solutions per iteration.
            max_iter: Maximum number of iterations.
            initial_variance: Initial variance passed into cma.
        """

        self.alphabet_size = data_utils.VOCAB_SIZE
        self.population_size = args.cmaes_population_size
        self.sequences_batch_size = args.n_chains
        self.initial_variance = args.cmaes_initial_variance


    def return_top_K(self, sequences, energies, K):
        """
        Over all generations in the history (List) find the Top K
        sequences and fitnesses and return them
        """
        values, indices = torch.topk(torch.stack(energies,0),K,0)
        return torch.stack(sequences,0)[indices], values


    def run(self, x, num_steps, energy_function, min_pos, max_pos, oracle, log_every=50):
        """Propose top `sequences_batch_size` sequences for evaluation."""
        with torch.no_grad():
            full_seq_len = x.size(1)
            preserve_left = x[0,:min_pos].clone()
            preserve_right = x[0,max_pos+1:].clone()

            state_energy, fitness = energy_function.get_energy(x)

            # convert x1,x2 to numpy arrays
            fitness_history = [fitness.detach()]
            energy_history = [state_energy.detach()]

            x0 = x[0,min_pos:max_pos+1].flatten().cpu().numpy()  # L*V
            
            seq_len = max_pos+1-min_pos

            def objective_function(soln):
                """
                soln is a [seq_len*vocab_size] numpy array
                """
                x_ = torch.from_numpy(soln).to(x.device)
                x_ = x_.view(seq_len, self.alphabet_size)
                
                x_ = torch.argmax(x_,-1)
                x_hard = torch.nn.functional.one_hot(x_, self.alphabet_size).to(x_.device).float()
                x_hard = torch.cat([preserve_left, x_hard, preserve_right],0).unsqueeze(0)  # [1,L,V]
                energy, fitness = energy_function.get_energy(x_hard)

                return -energy

            # Starting solution gives equal weight to all residues at all positions
            opts = {"popsize": self.population_size, "verbose": -9, "verb_log": 0}

            
            # x0 is a L*V flattened one-hot
            es = cma.CMAEvolutionStrategy(x0, np.sqrt(self.initial_variance), opts)

            # Explore until we reach `self.max_iter` or run out of model queries
            seq_history = []
            fit_history = []
            for step in range(num_steps):
                # `ask_and_eval` generates a new population of sequences
                solutions, fitnesses = es.ask_and_eval(objective_function)
                fitnesses = np.stack([f.cpu().numpy() for f in fitnesses])
                # `tell` updates model parameters
                es.tell(solutions, fitnesses)

                # Store scores of generated sequences
                for soln, f in zip(solutions, fitnesses):
                    x_ = torch.from_numpy(soln).to(x.device).float()
                    x_ = x_.view(seq_len, self.alphabet_size)
                    x_ = torch.argmax(x_,-1)
                    x_hard = torch.nn.functional.one_hot(x_, self.alphabet_size).to(x_.device).float()
                    x_hard = torch.cat([preserve_left, x_hard, preserve_right],0)  # [L,V]

                    seq_history += [x_hard.float()]
                    fit_history += [torch.from_numpy(-f).float()]

                if step>0 and (step+1) % log_every == 0:

                    top_K_seqs, top_K_energies = self.return_top_K(seq_history, fit_history, self.sequences_batch_size)
                    top_K_seqs = top_K_seqs.view(self.sequences_batch_size, full_seq_len, self.alphabet_size)
                    fitness_history += [energy_function.get_fitness(top_K_seqs).to(x.device)]
                    energy_history += [ top_K_energies[:,0].to(x.device) ]
                    # save mem and time by re-init seq_history to best K so far
                    seq_history = [ tk[0] for tk in torch.chunk(top_K_seqs, self.sequences_batch_size) ]
                    fit_history = [ fk[0] for fk in torch.chunk(top_K_energies, self.sequences_batch_size) ]

                    energy_quantiles = np.quantile(energy_history[-1].cpu().numpy(), [0.5,0.9])
                    fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
                    gt_score_quantiles = np.quantile(oracle(top_K_seqs).cpu().numpy(), [0.5, 0.9])

                    print(f'[Iteration {step}] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
                    print(f'[Iteration {step}] pred fitness 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
                    print(f'[Iteration {step}] oracle fitness 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
                    print('', flush=True)


            top_K_seqs, top_K_energies = self.return_top_K(seq_history, fit_history, self.sequences_batch_size)
            top_K_seqs = top_K_seqs.view(self.sequences_batch_size, full_seq_len, self.alphabet_size)
            best_fitness = energy_function.get_fitness(top_K_seqs)
            # best predicted samples - torch.Tensor
            # best predicted energy - numpy
            # best predicted fitness - numpy
            # all predicted energy
            # all predicted fitness
            # random_traj
            return top_K_seqs.to(x.device), top_K_energies.cpu().numpy(), best_fitness.cpu().numpy(), \
                torch.stack(energy_history).cpu().numpy(), torch.stack(fitness_history).cpu().numpy(), None
