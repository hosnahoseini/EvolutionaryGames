from player import Player
import numpy as np
from config import CONFIG
import copy as cp

class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def add_gaussian_noise(self, array):
        threshold = 0.2
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.normal(size=array.shape)

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        self.add_gaussian_noise(child.nn.W1)
        self.add_gaussian_noise(child.nn.W2)
        self.add_gaussian_noise(child.nn.b1)
        self.add_gaussian_noise(child.nn.b2)

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
            children = None
            parents = None

            if method_choose_parents == "top-k":
                prev_players = sorted(prev_players, key=lambda x: x.fitness, reverse=True)
                parents = prev_players[:num_players]
            elif method_choose_parents == "roulette wheel":
                parents = roulette_wheel_selection(prev_players, num_players)

            if method_repopulate == "simple":
                children = cp.deepcoopy(parents)
                children = map(lambda child: self.mutate(child), children)
            elif method_repopulate == "crossover":
                children = []
                for i in range(0, len(parents) - 1, 2):
                    child1, child2 = self.reproduction(parents[i], parents[i + 1])
                    children.append(child1)
                    children.append(child2)
            
            return children

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects

        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting
        method = "roulette wheel"
        if method == "top-k":
            players = sorted(players, key=lambda x: x.fitness, reverse=True)
            return players[: num_players]
        elif method == "roulette wheel":
            res = roulette_wheel_selection(players, num_players)
            return res
        

        # plotting
        fitness_list = [player.fitness for player in players]
        max_fitness = float(np.max(fitness_list))
        mean_fitness = float(np.mean(fitness_list))
        min_fitness = float(np.min(fitness_list))
        save_result(min_fitness, max_fitness, mean_fitness)


def roulette_wheel_selection(players, parent_numbers):
    
    # # Computes the totallity of the population fitness
    population_fitness = sum([chromosome.fitness for chromosome in players])
    
    # Computes for each chromosome the probability 
    probabilities = [chromosome.fitness / population_fitness for chromosome in players]
    
    # # Selects n chromosome based on the computed probabilities
    return np.random.choice(players, parent_numbers, p=probabilities).tolist()