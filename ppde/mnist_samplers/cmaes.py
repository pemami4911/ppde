
import cma
from typing import Optional, Tuple
from typing import List, Union
import numpy as np
import pandas as pd
import torch
from src.base_sampler import BaseSampler


class CMAES(BaseSampler):
    """
    An explorer which implements the covariance matrix adaptation evolution
    strategy (CMAES).
    Optimizes a continuous relaxation of the one-hot sequence that we use to
    construct a normal distribution around, sample from, and then argmax to get
    sequences for the objective function.
    http://blog.otoro.net/2017/10/29/visual-evolution-strategies/ is a helpful guide.
    """

    # def __init__(
    #     self,
    #     model: flexs.Model,
    #     rounds: int,
    #     sequences_batch_size: int,
    #     model_queries_per_batch: int,
    #     starting_sequence: str,
    #     alphabet: str,
    #     population_size: int = 15,
    #     max_iter: int = 400,
    #     initial_variance: float = 0.2,
    #     log_file: Optional[str] = None,
    # ):
    def __init__(self, args):
        """
        Args:
            population_size: Number of proposed solutions per iteration.
            max_iter: Maximum number of iterations.
            initial_variance: Initial variance passed into cma.
        """
        # super().__init__(
        #     model,
        #     name,
        #     rounds,
        #     sequences_batch_size,
        #     model_queries_per_batch,
        #     starting_sequence,
        #     log_file,
        # )

        self.alphabet = "01"
        self.population_size = args.cmaes_population_size
        self.sequences_batch_size = args.n_chains
        self.initial_variance = args.cmaes_initial_variance


    def propose_sequences(self, max_iter, model, oracle, x1, x2, log_every):
        """Propose top `sequences_batch_size` sequences for evaluation."""
        # measured_sequence_dict = dict(
        #     zip(measured_sequences["sequence"], measured_sequences["true_score"])
        # )

        # Keep track of new sequences generated this round
        #top_idx = measured_sequences["true_score"].argmax()
        #top_seq = measured_sequences["sequence"].to_numpy()[top_idx]
        #top_val = measured_sequences["true_score"].to_numpy()[top_idx]
        
        #sequences = {top_seq: top_val}
        state_energy, fitness = model.get_energy(x2, x1=x1)
        print(f'initial energy = {state_energy}')
        # convert x1,x2 to numpy arrays
        x1_single = x1[0].unsqueeze(0)
        fitness_history = [fitness]
        energy_history = [state_energy.cpu().numpy()]
        gt_fitness = [oracle(x1, x2)]
        x0 = x2[0].flatten().cpu().numpy()
        seq_len = len(x0)

        def objective_function(soln):
            """
            soln is a [seq_len*vocab_size] numpy array
            """
            #seq = self._soln_to_string(soln)

            # if seq in sequences:
            #     return sequences[seq]
            # if seq in measured_sequence_dict:
            #     return measured_sequence_dict[seq]
            x2 = torch.from_numpy(soln).to(x1.device)
            x2 = x2.view(seq_len,len(self.alphabet))
            x2 = torch.argmax(x2,-1).unsqueeze(0)
            logp, digitsum = model.get_energy(x2.float(), x1=x1_single)
            return -logp

        # Starting solution gives equal weight to all residues at all positions
        #x0 = self.string_to_one_hot(top_seq, self.alphabet).flatten()
        opts = {"popsize": self.population_size, "verbose": -9, "verb_log": 0}

        x0_ = np.zeros((seq_len, len(self.alphabet)))
        x0_[np.arange(seq_len),x0.astype('int')] = 1
        x0_ = x0_.flatten()
        
        es = cma.CMAEvolutionStrategy(x0_, np.sqrt(self.initial_variance), opts)

        # Explore until we reach `self.max_iter` or run out of model queries
        #initial_cost = model.cost
        history = []
        for step in range(max_iter):
            # `ask_and_eval` generates a new population of sequences
            solutions, fitnesses = es.ask_and_eval(objective_function)
            fitnesses = np.stack([f.cpu().numpy() for f in fitnesses])
            # `tell` updates model parameters
            es.tell(solutions, fitnesses)

            # Store scores of generated sequences
            history += [{'sequences': [], 'fitnesses': []}]
            for soln, f in zip(solutions, fitnesses):
                #sequences[self._soln_to_string(soln)] = f
                x2 = torch.from_numpy(soln).float()
                x2 = x2.view(seq_len, len(self.alphabet))
                x2 = torch.argmax(x2,-1)

                history[-1]['sequences'] += [x2.float()]
                history[-1]['fitnesses'] += [-f]
            #if step % 10 == 0:
            #    print(history[-1]['fitnesses'])
            if step>0 and (step+1) % log_every == 0:
                t = self.sequences_batch_size // self.population_size

                x2_new = torch.cat([ torch.stack(history[-i]['sequences'],0) for i in range(1,t+1) ],0)
                gt_fitness += [oracle(x1, x2_new.to(x1.device))]
                fitness_history += [model.get_fitness(x2_new.to(x1.device), x1=x1)]
                energy_history += [ np.concatenate([ np.stack(history[-i]['fitnesses'])[:,0] for i in range(1,t+1)] ) ]
                print(f'step {step}, energy = {np.mean(energy_history[-1])}')

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        t = self.sequences_batch_size // self.population_size
        new_seqs = np.stack([ torch.stack(history[-i]['sequences'],0).cpu().numpy() for i in range(1,t+1)])
                # Negate `objective_function` scores
        #preds = history[-1]['fitnesses']
        #sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        #return new_seqs[sorted_order], preds[sorted_order]
        return new_seqs, np.stack(energy_history, 0), \
                torch.stack(fitness_history, 0).cpu().numpy(), torch.stack(gt_fitness,0).cpu().numpy(), None

    def run(self, initial_population, num_steps, energy_function, oracle, logger, log_every=50):
        with torch.no_grad():
            seq_len = initial_population.size(1) // 2
            x1 = initial_population[:,:seq_len]
            x2 = initial_population[:,seq_len:]

            return self.propose_sequences(num_steps, energy_function, oracle, x1, x2, log_every)