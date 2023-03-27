import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.base_sampler import BaseSampler
import time


class PPDE(BaseSampler):
    def __init__(self, args):
        super().__init__()
        self.ppde_temp = 2  # optimal for locally balanced function g(t) = \sqrt(t)
        self.ppde_gwg_samples = args.ppde_gwg_samples
        self.ppde_pas_length = args.ppde_pas_length    

    def delta(self, x):
        wx = -(2. * x - 1)
        return wx
    
    def approximate_energy_change(self, wx, grad_x):
        return grad_x * wx.detach() / self.ppde_temp
    
    def run(self, initial_population, num_steps, energy_function, oracle, logger, log_every=50):
        """
        initial population is of shape [population_size, 2*sequence_length, vocab_size]
        """
        seq_len = initial_population.size(1) // 2
        x1 = initial_population[:,:seq_len]
        x2 = initial_population[:,seq_len:]
        n_chains = initial_population.size(0)
        random_idx = np.random.randint(0,n_chains)
        
        x1 = x1.detach()
        state_energy, fitness = energy_function.get_energy(x2, x1=x1)

        #seq_history = [state_seqs]
        fitness_history = [fitness.detach()]
        energy_history = [state_energy.detach()]
        random_traj = [x2[random_idx].view(28,28,1).detach().cpu().numpy()]
        gt_fitness = [oracle(x1, x2).detach()]

        fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
        gt_score_quantiles = np.quantile(gt_fitness[-1].cpu().numpy(), [0.5, 0.9])
        print(f'[Iteration 0] energy: 50% {np.median(energy_history[-1].cpu().numpy()):.3f}, 100% {np.max(energy_history[-1].cpu().numpy()):.3f}')        
        print(f'[Iteration 0] pred sum 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
        print(f'[Iteration 0] oracle sum 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
        print('')

        for i in range(num_steps):
            start_step = time.time()

            Delta = []
            Idx = []
            forward_categoricals = []

            start = time.time()
            x2 = x2.requires_grad_()
            current_energy, _ = energy_function.get_energy(x2, x1=x1)
            time_forward = time.time() - start
            # compute approximate energy change
            #grad_x = self.grad(x_cur, cur_fitness.sum())
            start = time.time()
            grad_x = torch.autograd.grad([current_energy.sum()], x2)[0]
            time_backward = time.time() - start
            delta = self.delta(x2)
            Delta += [delta]
            approx_forward_energy_change = self.approximate_energy_change(delta, grad_x)
            
            # Compute PIP
            cd_forward = torch.distributions.one_hot_categorical.OneHotCategorical(logits=approx_forward_energy_change)
            forward_categoricals += [cd_forward]
            # sample from the PIP. If GWG with multi samples, n_samples > 1
            if self.ppde_pas_length > 0:
                n_samples = 1
            else:    
                n_samples = np.random.randint(1, 2 * self.ppde_gwg_samples)
            # since cd_forward is a OneHotCategorical, this has shape [n_samples,batch_size,784]
            changes_all = cd_forward.sample((n_samples,))
            Idx += [changes_all]
            changes = (changes_all.sum(0) > 0.).float()
            # flip state
            x2_proposal = (1. - x2) * changes + x2 * (1. - changes)
            
            # If using PAS, continue to sample to obtain better proposal
            # ...do U intermediate steps
            if self.ppde_pas_length > 0:
                U = torch.randint(1, 2 * self.ppde_pas_length, size=(n_chains,1))
                max_u = torch.max(U).item()
                u_mask = torch.arange(max_u).expand(n_chains, max_u) < U
                u_mask = u_mask.float().to(x2_proposal.device)
                for step in range(1,max_u):                
                    delta = self.delta(x2_proposal)
                    Delta += [delta]
                    approx_forward_energy_change = self.approximate_energy_change(delta, grad_x)
                    pas_prob = torch.distributions.one_hot_categorical.OneHotCategorical(logits=approx_forward_energy_change)
                    forward_categoricals += [pas_prob]
                    changes_all = pas_prob.sample((1,))
                    changes = (changes_all.sum(0) > 0.).float()
                    new_x2_proposal = (1. - x2_proposal) * changes + x2_proposal * (1. - changes)
                    cur_u_mask = u_mask[:, step].unsqueeze(-1)
                    x2_proposal = cur_u_mask * new_x2_proposal + (1 - cur_u_mask) * x2_proposal
                    Idx += [changes_all]

                # last step
                delta = self.delta(x2_proposal)
                Delta += [delta]
                x2_proposal = x2_proposal.requires_grad_()
                proposal_energy, proposal_fitness = energy_function.get_energy(x2_proposal, x1=x1)
                #reverse_grad_x = self.grad(x_delta, proposed_fitness.sum())
                reverse_grad_x = torch.autograd.grad([proposal_energy.sum()], x2_proposal)[0]
                approx_reverse_energy_changes = self.approximate_energy_change(torch.stack(Delta[1:], dim=0), reverse_grad_x)
                log_ratio = 0
                for id in range(len(Idx)):
                    cd_reverse = torch.distributions.one_hot_categorical.OneHotCategorical(logits=approx_reverse_energy_changes[id])
                    log_ratio += u_mask[:,id] * (cd_reverse.log_prob(Idx[id]) - forward_categoricals[id].log_prob(Idx[id]))
                
            else:
                # last step for GWG
                x2_proposal = x2_proposal.requires_grad_()
                proposal_energy, proposal_fitness = energy_function.get_energy(x2_proposal, x1=x1)
                #reverse_grad_x = self.grad(x_delta, proposed_fitness.sum())
                reverse_grad_x = torch.autograd.grad([proposal_energy.sum()], x2_proposal)[0]
                reverse_delta = self.delta(x2_proposal)
                approximate_reverse_energy_change = self.approximate_energy_change(reverse_delta, reverse_grad_x)
                cd_reverse = torch.distributions.one_hot_categorical.OneHotCategorical(logits=approximate_reverse_energy_change)
                lp_forward = cd_forward.log_prob(Idx[0]).sum(0)                                    
                lp_reverse = cd_reverse.log_prob(Idx[0]).sum(0)
                log_ratio = lp_reverse - lp_forward
                
            m_term = (proposal_energy.squeeze() - current_energy.squeeze())
            la = m_term + log_ratio
            a = (la.exp() > torch.rand_like(la)).float()
            a = a.view(x2.size(0), 1)
            x2 = x2_proposal * a + x2 * (1. - a)
            new_energy = proposal_energy * a.squeeze() + current_energy * (1. - a).squeeze()
            random_traj += [x2[random_idx].view(28,28,1).detach().cpu().numpy()]
            fitness = proposal_fitness * a.squeeze() + fitness * (1. - a).squeeze()
            
            x2=x2.detach()
            new_energy = new_energy.detach()
            fitness = fitness.detach()
            end_step = time.time() - start_step

            if i > 0 and (i+1) % log_every == 0:
                gt_fitness += [oracle(x1, x2).detach()]
                fitness_history.append(fitness.detach())
                energy_history.append(new_energy.detach())

                fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
                gt_score_quantiles = np.quantile(gt_fitness[-1].cpu().numpy(), [0.5, 0.9])

                print(f'[Iteration {i}] energy: 50% {np.median(energy_history[-1].cpu().numpy()):.3f}, 100% {np.max(energy_history[-1].cpu().numpy()):.3f}')
                print(f'[Iteration {i}] pred sum 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
                print(f'[Iteration {i}] oracle sum 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
                print(f'[Iteration {i}] time frwd: {time_forward:.3f} s, time bckwd: {time_backward:.3f} s, time step: {end_step:.3f} s')
                print('')
            
        # final_population
        # energy history
        # sums
        # oracle history
        # random_traj
        return x2.cpu().numpy(), torch.stack(energy_history, 0).cpu().numpy(), \
                torch.stack(fitness_history, 0).cpu().numpy(), torch.stack(gt_fitness,0).cpu().numpy(), random_traj
