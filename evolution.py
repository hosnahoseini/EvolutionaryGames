from player import Player
import numpy as np
from config import CONFIG
import copy as cp
import json

class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.generation_cnt = 0

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

    def roulette_wheel_selection(self, players, parent_numbers):
        
        # # Computes the totallity of the population fitness
        population_fitness = sum([chromosome.fitness for chromosome in players])
        
        # Computes for each chromosome the probability 
        probabilities = [chromosome.fitness / population_fitness for chromosome in players]
        
        # # Selects n chromosome based on the computed probabilities
        return np.random.choice(players, parent_numbers, p=probabilities).tolist()

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
                parents =self.roulette_wheel_selection(prev_players, num_players)

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
            res = players[: num_players]
        elif method == "roulette wheel":
            res = self.roulette_wheel_selection(players, num_players)

        # plotting
        fitness_list = [player.fitness for player in players]
        max_fitness = float(np.max(fitness_list))
        mean_fitness = float(np.mean(fitness_list))
        min_fitness = float(np.min(fitness_list))
        self.save_result(min_fitness, max_fitness, mean_fitness)

        return res

    def save_result(self, min_fitness, max_fitness, mean_fitness):
            if self.generation_cnt == 0:
                fitness_results = {
                    'min_fitness': [min_fitness],
                    'max_fitness': [max_fitness],
                    'mean_fitness': [mean_fitness]
                }
                with open('fitness_data.json', 'w') as out_file:
                    json.dump(fitness_results, out_file)
            else:
                with open('fitness_data.json', 'r') as in_file:
                    fitness_results = json.load(in_file)

                fitness_results['min_fitness'].append(min_fitness)
                fitness_results['max_fitness'].append(max_fitness)
                fitness_results['mean_fitness'].append(mean_fitness)

                with open('fitness_data.json', 'w') as out_file:
                    json.dump(fitness_results, out_file)

            self.generation_cnt += 1