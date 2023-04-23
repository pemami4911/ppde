import torch
import numpy as np
import random
from ppde.base_sampler import BaseSampler
from ppde.third_party.hsu import data_utils
from ppde.metrics import n_hops

class RandomSampler(BaseSampler):
    def __init__(self, args):
        self.T_max = args.simulated_annealing_temp
        self.T = self.T_max
        self.muts_per_seq_param = args.muts_per_seq_param # 1.5
        self.decay_rate = args.decay_rate
        self.AA_idxs = [i for i in range(data_utils.VOCAB_SIZE)]
    
    def make_n_random_edits(self,seq, nedits, min_pos=None, max_pos=None):
        seq = seq.squeeze()
        if min_pos is None:
            min_pos = 0
        
        if max_pos is None:
            max_pos = seq.size(0)
        # Create non-redundant list of positions to mutate.
        l = list(range(min_pos,max_pos+1))
        nedits = min(len(l), nedits)
        random.shuffle(l)
        # pick n random positions
        pos_to_mutate = l[:nedits]    
        
        for i in range(nedits):
            # random mutation
            pos = pos_to_mutate[i]
            cur_AA = torch.argmax(seq[pos]).item()
            candidates = list(set(self.AA_idxs) - set([cur_AA]))
            seq[pos][cur_AA] = 0
            seq[pos][np.random.choice(candidates)] = 1
            
        return seq.unsqueeze(0) #.view(1,h*w)


    def propose_seqs(self,seqs, mu_muts_per_seq, min_pos, max_pos):
        mseqs = []
        for i,s in enumerate(seqs):
            n_edits = torch.poisson(mu_muts_per_seq[i]-1) + 1
            mseqs.append(self.make_n_random_edits(s, n_edits.int().item(), min_pos, max_pos)) 
        return mseqs


    def run(self, initial_population, num_steps, energy_function, min_pos, max_pos, oracle, log_every=50):

        with torch.no_grad():
            n_chains = initial_population.size(0)
            seq_len = initial_population.size(1)
            x = initial_population
            random_idx = np.random.randint(0,n_chains)
            mu_muts_per_seq = torch.from_numpy(self.muts_per_seq_param * np.random.rand(n_chains) + 1)

            # convert population to List of tensors
            state_energy, fitness = energy_function.get_energy(x)
            state_seqs = torch.chunk(x.clone(),n_chains)
            #seq_history = [state_seqs]
            fitness_history = [fitness]
            energy_history = [state_energy.clone()]
            random_traj = [state_seqs[random_idx].cpu().numpy()]
            gt_fitness = oracle(x)
            all_x = [x.detach().cpu().numpy()]

            fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
            gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])
            energy_quantiles = np.quantile(state_energy.cpu().numpy(), [0.5,0.9])

            print(f'[Iteration 0] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
            print(f'[Iteration 0] pred fitness 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
            print(f'[Iteration 0] oracle fitness 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
            print('')

            for i in range(num_steps):
                
                #current_energy = energy_history[-1]
               
                # random proposals
                proposal_seqs = self.propose_seqs(state_seqs, mu_muts_per_seq, min_pos, max_pos)
                x_proposal = torch.cat(proposal_seqs)           
                proposal_energy, proposal_fitness = energy_function.get_energy(x_proposal)

                x_new = x_proposal 
                
                # resets
                state_seqs = torch.chunk(x.clone(), n_chains)
                
                #aprob = aprob.view(n_chains)
               
                # replace -inf with 0
                proposal_energy[torch.isneginf(proposal_energy)] = 0
                proposal_fitness[torch.isneginf(proposal_fitness)] = 0

                state_energy = proposal_energy# * aprob + current_energy * (1. - aprob)
                energy_history += [state_energy.clone()]
                fitness = proposal_fitness #* aprob + fitness * (1. - aprob)
                fitness_history += [fitness.detach()]
                all_x += [x_new.detach().cpu().numpy()]

                random_traj += [state_seqs[random_idx].cpu().numpy()]
                self.T = self.T_max * self.decay_rate**i
                
                if i > 0 and (i+1) % log_every == 0:
                    gt_fitness = oracle(x_new)
                    #fitness_history += [fitness]
                    energy_quantiles = np.quantile(state_energy.cpu().numpy(), [0.5,0.9])
                    fitness_quantiles = np.quantile(fitness.cpu().numpy(), [0.5,0.9])
                    gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])
                    mean_hops, std_hops = n_hops(x_new, torch.from_numpy(data_utils.seqs_to_onehot(oracle.potts.wtseqs)[0]).to(x.device).float())

                    print(f'[Iteration {i}] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
                    print(f'[Iteration {i}] pred fitness 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
                    print(f'[Iteration {i}] oracle fitness 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
                    print(f'[Iteration {i}] mean hops = {mean_hops:.2f}, std hops = {std_hops:.2f}')
                    
                    print('',flush=True)

            energy_history = torch.stack(energy_history)  # [num_steps,num_chains]
            best_energy, best_idxs = torch.max(energy_history,0)

            all_x = np.stack(all_x,0)

            #all_x = torch.stack(all_x,0)  # [num_steps,...]
            best_x = np.stack([all_x[best_idxs[i],i] for i in range(n_chains)],0)
            fitness_history = torch.stack(fitness_history, 0)
            best_fitness = torch.stack([fitness_history[best_idxs[i],i] for i in range(n_chains)],0)
            # best predicted samples - torch.Tensor
            # best predicted energy - numpy
            # best predicted fitness - numpy
            # all predicted energy
            # all predicted fitness
            # random_traj
            return torch.from_numpy(best_x).to(initial_population.device), best_energy.cpu().numpy(), best_fitness.cpu().numpy(), \
                energy_history.cpu().numpy(), fitness_history.cpu().numpy(), random_traj
