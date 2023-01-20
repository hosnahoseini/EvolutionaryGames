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

    def add_gaussian_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.normal(size=array.shape)
            
    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        threshold = 0.2

        self.add_gaussian_noise(child.nn.W1, threshold)
        self.add_gaussian_noise(child.nn.W2, threshold)
        self.add_gaussian_noise(child.nn.b1, threshold)
        self.add_gaussian_noise(child.nn.b2, threshold)

    def crossover(self, child1_array, child2_array, parent1_array, parent2_array):
        row_size, column_size = child1_array.shape
        break_point = int(row_size / 2)

        random_number = np.random.uniform(0, 1, 1)
        if random_number > 0.5:
            child1_array[:break_point, :] = parent1_array[:break_point:, :]
            child1_array[break_point:, :] = parent2_array[break_point:, :]

            child2_array[:break_point, :] = parent2_array[:break_point:, :]
            child2_array[break_point:, :] = parent1_array[break_point:, :]
        else:
            child1_array[:break_point, :] = parent2_array[:break_point:, :]
            child1_array[break_point:, :] = parent1_array[break_point:, :]

            child2_array[:break_point, :] = parent1_array[:break_point:, :]
            child2_array[break_point:, :] = parent2_array[break_point:, :]

    def reproduction(self, parent1, parent2):
        child1 = Player(self.mode)
        child2 = Player(self.mode)

        self.crossover(child1.nn.W1, child2.nn.W1, parent1.nn.W1, parent2.nn.W1)
        self.crossover(child1.nn.W2, child2.nn.W2, parent1.nn.W2, parent2.nn.W2)
        self.crossover(child1.nn.b1, child2.nn.b1, parent1.nn.b1, parent2.nn.b1)
        self.crossover(child1.nn.b2, child2.nn.b2, parent1.nn.b2, parent2.nn.b2)

        self.mutate(child1)
        self.mutate(child2)
        return child1, child2

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


            # num_players example: 150
            # prev_players: an array of `Player` objects
            method_choose_parents = "roulette wheel"
            method_repopulate = "crossover"
            children = []
            parents = []

            method = "top-k"
            switch(method_choose_parents){
                case "top-k":
                    players.sort(players, key=lambda x: x.fitness, reverse=True)
                    parents = players[:num_players]
                    break
                case "roulette wheel":
                    parents = roulette_wheel_selection(players, num_players)
                    break
            }

            switch(method_repopulate){
                case "simple":
                    children = cp.deepcoopy(parents)
                    children = map(lambda child: elf.mutate(child), children)
                    break;
                case "crossover":
                    for i in range(0, len(parents), 2):
                        child1, child2 = self.reproduction(parents[i], parents[i + 1])
                        children.append(child1)
                        children.append(child2)
                    break;
            }
            
            return children

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
                return roulette_wheel_selection(players, num_players)
        }

    def roulette_wheel_selection(population, n):
    
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome.fitness for chromosome in population])
        
        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome.fitness / population_fitness for chromosome in population]
        
        # Selects one chromosome based on the computed probabilities
        return np.random.choice(population, n, p=chromosome_probabilities), 