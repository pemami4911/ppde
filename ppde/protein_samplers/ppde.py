import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.base_sampler import BaseSampler
import time
from src.utils import mut_distance, mutation_mask


class PPDE_PAS(BaseSampler):
    def __init__(self, args):
        super().__init__()
        self.ppde_temp = 2  # locally balanced function g(t) = \sqrt(t)
        self.ppde_pas_length = args.ppde_pas_length    
        self.nmut_threshold = args.nmut_threshold
        if self.nmut_threshold == 0:
            # big num
            self.nmut_threshold = np.iinfo(np.int32).max


    def approximate_energy_change(self, score_change):
        return score_change / self.ppde_temp
    
    def run(self, initial_population, num_steps, energy_function, min_pos, max_pos, oracle, log_every=50):
        """
        initial_population is [n_chains, seq_len, vocab_size]

        Largely inspired by https://github.com/ha0ransun/Path-Auxiliary-Sampler/blob/a93912beda8e264f04704180e505a1b333f227c8/PAS/debm/sampling/multistep_sampler.py#L167

        """
        print(min_pos, max_pos)

        n_chains = initial_population.size(0)
        seq_len = initial_population.size(1)
        x = initial_population.clone()
        x_rank = len(x.shape)-1
        random_idx = np.random.randint(0,n_chains)
        all_x = [x.detach().cpu().numpy()]

        x = x.detach()
        with torch.no_grad():
            state_energy, fitness = energy_function.get_energy(x)

        #seq_history = [state_seqs]
        fitness_history = [fitness.detach()]
        energy_history = [state_energy.detach()]
        random_traj = [x[random_idx].detach().cpu().numpy()]
        gt_fitness = oracle(x).detach()

        fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
        gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])
        energy_quantiles = np.quantile(energy_history[-1].cpu().numpy(), [0.5,0.9])
        
        print(f'[Iteration 0] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}')
        print(f'[Iteration 0] pred fit 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
        print(f'[Iteration 0] oracle fit 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
        print('')

        cur_x = x.clone()
        pos_mask = torch.ones_like(cur_x).to(cur_x.device)
        pos_mask[:,min_pos:max_pos+1] = 0
        pos_mask = pos_mask.bool()
        pos_mask = pos_mask.reshape(n_chains,-1)

        for i in range(num_steps):
            #start_step = time.time()
            ###### sample path length for PAS
            U = torch.randint(1, 2 * self.ppde_pas_length, size=(n_chains,1))
            max_u = torch.max(U).item()
            u_mask = torch.arange(max_u).expand(n_chains, max_u) < U
            u_mask = u_mask.float().to(x.device)

            onehot_Idx = []
            traj_list = []
            forward_categoricals = []

            #start = time.time()
            cur_x = cur_x.requires_grad_()
            #current_energy, current_fitness = energy_function.get_energy(cur_x)
            #time_forward = time.time() - start
            #start = time.time()
            #grad_x = torch.autograd.grad([current_energy.sum()], cur_x)[0]
            #time_backward = time.time() - start
            current_energy, current_fitness, grad_x = energy_function.get_energy_and_grads(cur_x)
                           
            # do U intermediate steps
            with torch.no_grad():
                for step in range(max_u):

                    # compute the mut_distance between cur_x and wt
                    dist = mut_distance(cur_x, energy_function.wt_onehot)
                    # if dist == threshold, only valid next mutations 
                    # are substitutions that reduce the mut_distance.
                    # set approx_forward_energy_change to -inf everywhere else. 
                    mask_flag = (dist == self.nmut_threshold).bool()
                    mask_flag = mask_flag.reshape(n_chains)
                    mask = mutation_mask(cur_x, energy_function.wt_onehot)
                    mask = mask.reshape(n_chains,-1)

                    # mask out min/max positions
                    
                    # Compute PIP
                    score_change = grad_x - (grad_x * cur_x).sum(-1).unsqueeze(-1)
                    traj_list += [cur_x]
                    approx_forward_energy_change = score_change.reshape(n_chains,-1) / self.ppde_temp
                    
                    # Apply mask to constrain proposals within edit distance of WT
                    mask[~mask_flag] = False
                    
                    approx_forward_energy_change[mask] = -np.inf
                    approx_forward_energy_change[pos_mask] = -np.inf

                    cd_forward = torch.distributions.one_hot_categorical.OneHotCategorical(logits=approx_forward_energy_change)
                    forward_categoricals += [cd_forward]
                    changes_all = cd_forward.sample((1,)).squeeze(0)
                    onehot_Idx += [changes_all]
                    changes_all = changes_all.view(n_chains, seq_len, -1)
                    row_select = changes_all.sum(-1).unsqueeze(-1)  # [n_chains,seq_len,1]
                    new_x = cur_x * (1.0 - row_select) + changes_all
                    cur_u_mask = u_mask[:, step].unsqueeze(-1).unsqueeze(-1)
                    cur_x = cur_u_mask * new_x + (1 - cur_u_mask) * cur_x

                    
                y = cur_x
            # last step
            y = y.requires_grad_()
            #proposed_energy, proposed_fitness = energy_function.get_energy(y)
            #grad_y = torch.autograd.grad(proposed_energy.sum(), y)[0].detach()
            proposed_energy, proposed_fitness, grad_y = energy_function.get_energy_and_grads(y)
            grad_y = grad_y.detach()

            with torch.no_grad():
                 # backwd from y -> x
                traj_list.append(y)
                traj = torch.stack(traj_list[1:], dim=1)
                reverse_score_change = grad_y.unsqueeze(1) - (grad_y.unsqueeze(1) * traj).sum(-1).unsqueeze(-1)
                reverse_score_change = reverse_score_change.reshape(n_chains, max_u, -1) / 2.0
                log_ratio = 0
                for id in range(len(onehot_Idx)):
                    cd_reverse = torch.distributions.one_hot_categorical.OneHotCategorical(logits=reverse_score_change[:,id])
                    log_ratio += u_mask[:,id] * (cd_reverse.log_prob(onehot_Idx[id]) - forward_categoricals[id].log_prob(onehot_Idx[id]))
                
                #log_acc = log_backwd - log_fwd
                m_term = (proposed_energy.squeeze() - current_energy.squeeze())
                log_acc = m_term + log_ratio
                
                accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
                new_x = y * accepted + (1.0 - accepted) * x
                
            cur_x = new_x
            new_energy = proposed_energy * accepted.squeeze() + current_energy * (1. - accepted.squeeze())
            random_traj += [cur_x[random_idx].detach().cpu().numpy()]
            fitness = proposed_fitness * accepted.squeeze() + current_fitness * (1. - accepted.squeeze())
            energy_history += [new_energy.detach()]
            fitness_history += [fitness.detach()]
            all_x += [cur_x.detach().cpu().numpy()]

            if i > 0 and (i+1) % log_every == 0:
                gt_fitness = oracle(cur_x).detach()

                fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
                gt_score_quantiles = np.quantile(gt_fitness.cpu().numpy(), [0.5, 0.9])
                energy_quantiles = np.quantile(energy_history[-1].cpu().numpy(), [0.5,0.9])

                print(f'[Iteration {i}] energy: 50% {energy_quantiles[0]:.3f}, 90% {energy_quantiles[1]:.3f}', flush=True)
                print(f'[Iteration {i}] pred 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}', flush=True)
                print(f'[Iteration {i}] oracle 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}', flush=True)
                print(f'   # accepted = {torch.sum(accepted)}')
                #print(f'[Iteration {i}] time frwd: {time_forward:.3f} s, time bckwd: {time_backward:.3f} s, time step: {end_step:.3f} s')
                print('', flush=True)
            
        energy_history = torch.stack(energy_history)  # [num_steps,num_chains]
        best_energy, best_idxs = torch.max(energy_history,0)
        
        all_x = np.stack(all_x,0)  # [num_steps,...]
        fitness_history = torch.stack(fitness_history, 0)

        if n_chains > 1:
            best_x = np.stack([all_x[best_idxs[i],i] for i in range(n_chains)],0)
            best_fitness = torch.stack([fitness_history[best_idxs[i],i] for i in range(n_chains)],0)
        else:
            best_x = all_x[best_idxs]
            best_fitness = fitness_history[best_idxs]

        # best predicted samples - torch.Tensor
        # best predicted energy - numpy
        # best predicted fitness - numpy
        # all predicted energy
        # all predicted fitness
        # random_traj
        return torch.from_numpy(best_x).to(initial_population.device), best_energy.cpu().numpy(), best_fitness.cpu().numpy(), \
            energy_history.cpu().numpy(), fitness_history.cpu().numpy(), random_traj