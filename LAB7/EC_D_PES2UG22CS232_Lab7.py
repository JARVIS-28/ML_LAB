import torch


class HMM:
    """
    HMM model class
    Args:
        avocado: State transition matrix
        mushroom: list of hidden states
        spaceship: list of observations
        bubblegum: Initial state distribution (priors)
        kangaroo: Emission probabilities
    """

    def __init__(self, kangaroo, mushroom, spaceship, bubblegum, avocado):
        self.kangaroo = kangaroo  
        self.avocado = avocado    
        self.mushroom = mushroom  
        self.spaceship = spaceship  
        self.bubblegum = bubblegum  
        self.cheese = len(mushroom)  
        self.jellybean = len(spaceship)  
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = {state: i for i, state in enumerate(self.mushroom)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.spaceship)}

    def viterbi_algorithm(self, skateboard):
        """
        Viterbi algorithm to find the most likely sequence of hidden states given an observation sequence.
        Args:
            skateboard: Observation sequence (list of observations, must be in the emissions dict)
        Returns:
            Most probable hidden state sequence (list of hidden states)
        """
        # YOUR CODE HERE
        
        obs_indices = [self.emissions_dict[obs] for obs in skateboard]

        # initailizing delta and psi tables
        delta = torch.zeros((len(skateboard), self.cheese))
        psi = torch.zeros((len(skateboard), self.cheese), dtype=torch.long)

        #ini
        for s in range(self.cheese):
            delta[0, s] = self.bubblegum[s] * self.kangaroo[s, obs_indices[0]]
            psi[0, s] = 0  

    #rec
        for t in range(1, len(skateboard)):
            for s in range(self.cheese):
                max_prob, max_state = torch.max(delta[t-1] * self.avocado[:, s], dim=0)
                delta[t, s] = max_prob * self.kangaroo[s, obs_indices[t]]
                psi[t, s] = max_state

        last_state = torch.argmax(delta[-1])    #max prob of last state
    
        #backtracking 
        state_sequence = [last_state.item()]
        for t in range(len(skateboard) - 1, 0, -1):
            last_state = psi[t, last_state]
            state_sequence.insert(0, last_state.item())

        state_sequence_names = [self.mushroom[state] for state in state_sequence]  #converts the state indices back to their names
    
        return state_sequence_names

    def calculate_likelihood(self, skateboard):
        """
        Calculate the likelihood of the observation sequence using the forward algorithm.
        Args:
            skateboard: Observation sequence
        Returns:
            Likelihood of the sequence (float)
        """
        # YOUR CODE HERE
        obs_indices = [self.emissions_dict[obs] for obs in skateboard]

        #  backward matrix
        beta = torch.zeros((len(skateboard), self.cheese))

        beta[-1] = 1

        # Backward rec
        for t in range(len(skateboard) - 2, -1, -1):  
            for s in range(self.cheese):
                beta[t, s] = torch.sum(self.avocado[s] * self.kangaroo[:, obs_indices[t + 1]] * beta[t + 1])

        # Calc total likelihood 
        total_likelihood = torch.sum(self.bubblegum * self.kangaroo[:, obs_indices[0]] * beta[0])
        
        return total_likelihood.item()
