import torch
import copy 
import numpy as np
import random
from src.base_sampler import BaseSampler


class SimulatedAnnealing(BaseSampler):
    def __init__(self, args):
        self.T_max = args.simulated_annealing_temp
        self.T = self.T_max
        self.muts_per_seq_param = args.muts_per_seq_param # 1.5
        self.decay_rate = args.decay_rate

    def acceptance_prob(self, e_proposal, e_current):
        ap = ((e_proposal - e_current) / self.T).exp()
        ap[ap > 1] = 1.
        return ap
    
    @staticmethod
    def make_n_random_edits(seq, nedits):
        #_,h,w = seq.shape
        seq = seq.squeeze()
        # Create non-redundant list of positions to mutate.
        l = list(range(seq.size(0)))
        nedits = min(len(l), nedits)
        random.shuffle(l)
        # pick n random positions
        pos_to_mutate = l[:nedits]    
        
        for i in range(nedits):
            pos = pos_to_mutate[i]
            # flip dim
            seq[pos] = 1 - seq[pos]
            
        return seq.unsqueeze(0) #.view(1,h*w)


    @staticmethod
    def propose_seqs(seqs, mu_muts_per_seq):
        mseqs = []
        for i,s in enumerate(seqs):
            n_edits = torch.poisson(mu_muts_per_seq[i]-1) + 1
            mseqs.append(SimulatedAnnealing.make_n_random_edits(s, n_edits.int().item())) 
        return mseqs


    def run(self, initial_population, num_steps, energy_function, oracle, logger, log_every=50):

        with torch.no_grad():
            seq_len = initial_population.size(1) // 2
            x1 = initial_population[:,:seq_len]
            x2 = initial_population[:,seq_len:]
            n_chains = initial_population.size(0)
            random_idx = np.random.randint(0,n_chains)
            mu_muts_per_seq = torch.from_numpy(self.muts_per_seq_param * np.random.rand(n_chains) + 1)

            # convert population to List of tensors
            state_energy, fitness = energy_function.get_energy(x2, x1=x1)
            state_seqs = torch.chunk(x2.clone(),n_chains)
            #seq_history = [state_seqs]
            fitness_history = [fitness]
            energy_history = [state_energy.clone()]
            random_traj = [state_seqs[random_idx].view(28,28,1).cpu().numpy()]
            gt_fitness = [oracle(x1, x2)]

            fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
            gt_score_quantiles = np.quantile(gt_fitness[-1].cpu().numpy(), [0.5, 0.9])

            print(f'[Iteration 0] energy: 50% {np.median(state_energy.cpu().numpy()):.3f}, 100% {np.max(state_energy.cpu().numpy()):.3f}')
            print(f'[Iteration 0] pred sum 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
            print(f'[Iteration 0] oracle sum 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
            print('')

            for i in range(num_steps):
                
                current_energy = energy_history[-1]
               
                # random proposals
                proposal_seqs = SimulatedAnnealing.propose_seqs(state_seqs, mu_muts_per_seq)
                # TODO: fix?
                x2_proposal = torch.cat(proposal_seqs)           
                proposal_energy, proposal_fitness = energy_function.get_energy(x2_proposal, x1=x1)
                
                # Make sequence acceptance/rejection decisions
                aprob = self.acceptance_prob(proposal_energy, current_energy)                
                aprob = (aprob > torch.rand_like(aprob)).float()
                aprob = aprob.view(n_chains,1)
                x2_new = x2_proposal * aprob + x2 * (1. - aprob)
                state_seqs = torch.chunk(x2_new.clone(), n_chains)
                aprob = aprob.view(n_chains)
                state_energy = proposal_energy * aprob + current_energy * (1. - aprob)
                energy_history.append(state_energy.clone())
                
                random_traj += [state_seqs[random_idx].view(28,28,1).cpu().numpy()]
                self.T = self.T_max * self.decay_rate**i
                
                if i > 0 and (i+1) % log_every == 0:
                    fitness = proposal_fitness * aprob + fitness * (1. - aprob)
                    gt_fitness += [oracle(x1, x2_new)]
                    fitness_history.append(fitness)
                    fitness_quantiles = np.quantile(fitness.cpu().numpy(), [0.5,0.9])
                    gt_score_quantiles = np.quantile(gt_fitness[-1].cpu().numpy(), [0.5, 0.9])

                    print(f'[Iteration {i}] energy: 50% {np.median(state_energy.cpu().numpy()):.3f}, 100% {np.max(state_energy.cpu().numpy()):.3f}')
                    print(f'[Iteration {i}] pred sum 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
                    print(f'[Iteration {i}] oracle sum 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
                    print('')

            # final_population
            # energy history
            # sums
            # oracle history
            # random_traj
            return torch.stack(state_seqs,0).cpu().numpy(), torch.stack(energy_history, 0).cpu().numpy(), \
                   torch.stack(fitness_history, 0).cpu().numpy(), torch.stack(gt_fitness,0).cpu().numpy(), random_traj