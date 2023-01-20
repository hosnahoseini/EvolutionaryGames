from player import Player
import numpy as np
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        pass


    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover

            new_players = prev_players
            return new_players

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        method = "top-k"
        switch(method){
            case "top-k":
                players.sort(players, key=lambda x: x.fitness, reverse=True)
                return players[: num_players]
            case "roulette wheel":
                roulette_wheel_selection(players, num_players)
        }

    def roulette_wheel_selection(population, n):
    
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome.fitness for chromosome in population])
        
        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome.fitness / population_fitness for chromosome in population]
        
        # Selects one chromosome based on the computed probabilities
        return np.random.choice(population, n, p=chromosome_probabilities), 