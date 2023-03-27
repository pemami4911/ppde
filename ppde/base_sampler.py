
class BaseSampler(AbstractBaseClass):
    """Base class for samplers
    """
    def run(self, initial_population, num_steps, energy_function, min_pos, max_pos, oracle):
        """
        Inputs
            initial_population: torch.Tensor of shape [population_size, sequence_length, vocab_size]
            num_steps: int
            energy_function: callable for getting score
            min_pos, max_pos: restricting mutations to a subsequence (min_pos,max_pos)
            oracle: ground truth model for scoring population
        Returns
            final_population: torch.Tensor of shape [population_size, sequence_length, vocab_size]
                if return_full_population==True, torch.Tensor of shape [num_steps, pop_size, sequence_length, vocab_size]
            energy_history: torch.Tensor of shape [num_steps, population_size]
            (Optional) sums: torch.Tensor of shape [num_steps, population_size]
            oracle_sums: torch.Tensor of shape [num_steps, population_size]
            random_trajectory: a List of torch.Tensor of sequence length shape [sequence_length, vocab_size]
                containing the full trajectory of evolved MNIST images for visualization
        """
        raise NotImplementedError