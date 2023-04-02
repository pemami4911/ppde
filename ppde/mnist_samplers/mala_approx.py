import torch
import matplotlib.pyplot as plt
import numpy as np
from ppde.base_sampler import BaseSampler


class MALAApprox(BaseSampler):
    def __init__(self, args):
        self.diffusion_relaxation_tau = args.diffusion_relaxation_tau
        self.diffusion_step_size = args.diffusion_step_size        

    @staticmethod
    def straight_through_sample(x_relaxed_discrete):
        x_soft = x_relaxed_discrete.rsample()
        return (x_soft + torch.round(x_soft)) - x_soft

    def run(self, initial_population, num_steps, energy_function, min_pos=0, max_pos=784, oracle=None, log_every=50):
    
        seq_len = initial_population.size(1) // 2
        x1 = initial_population[:,:seq_len].detach()
        x2 = initial_population[:,seq_len:]
        n_chains = initial_population.size(0)
        random_idx = np.random.randint(0,n_chains)
        
        x2_soft = (1 - self.diffusion_relaxation_tau) * (0.5 * torch.ones_like(x2)) + self.diffusion_relaxation_tau * x2
        x2_relaxed_discrete = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.Tensor([self.diffusion_relaxation_tau]).to(x2.device), probs=x2_soft)    
        x2_logits = x2_relaxed_discrete.logits

        x2_hard = x2_relaxed_discrete.sample()
        state_energy, fitness = energy_function.get_energy(x2_hard, x1=x1)
        energy_history = [state_energy.detach()]
        fitness_history = [fitness.detach()]
        random_traj = [x2_hard[random_idx].view(28,28,1).detach().cpu().numpy()]
        gt_fitness = [oracle(x1, x2_hard).detach()]

        fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
        gt_score_quantiles = np.quantile(gt_fitness[-1].cpu().numpy(), [0.5, 0.9])
        print(f'[Iteration 0] pred sum 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
        print(f'[Iteration 0] oracle sum 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
        print('')

        for i in range(num_steps):

            x2_logits = x2_logits.requires_grad_()
            x2_relaxed_discrete = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                                  torch.Tensor([self.diffusion_relaxation_tau]).to(x2.device), logits=x2_logits) 
   
            x2_hard_differentiable = MALAApprox.straight_through_sample(x2_relaxed_discrete)
            energy, fitness = energy_function.get_energy(x2_hard_differentiable, x1=x1)
            
            x2_grad = torch.autograd.grad([energy.sum()], [x2_logits])[0]
            x2_logits = torch.distributions.normal.Normal(x2_logits + (self.diffusion_step_size/2.) * x2_grad,
                                                        self.diffusion_step_size ** 2).sample()
            
            fitness = fitness.detach()
            
            # latent back to image
            x2 = x2_hard_differentiable
            random_traj += [x2[random_idx].view(28,28,1).detach().cpu().numpy()]
            
            energy_history.append(energy.detach())

            if i > 0 and (i+1) % log_every == 0:
                gt_fitness += [oracle(x1, x2).detach()]
                fitness_history.append(fitness)
                fitness_quantiles = np.quantile(fitness_history[-1].cpu().numpy(), [0.5,0.9])
                gt_score_quantiles = np.quantile(gt_fitness[-1].cpu().numpy(), [0.5, 0.9])
              
                print(f'[Iteration {i}] pred sum 50% {fitness_quantiles[0]:.3f}, 90% {fitness_quantiles[1]:.3f}')
                print(f'[Iteration {i}] oracle sum 50% {gt_score_quantiles[0]:.3f}, 90% {gt_score_quantiles[1]:.3f}')
                print('')
        
        # final_population
        # energy history
        # sums
        # oracle history
        # random_traj
        return x2.detach().cpu().numpy(), torch.stack(energy_history, 0).cpu().numpy(), \
            torch.stack(fitness_history, 0).cpu().numpy(), \
               torch.stack(gt_fitness,0).cpu().numpy(), random_traj