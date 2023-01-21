import matplotlib.pyplot as plt
import json

with open('fitness_data.json', 'r') as f:
    fitness_results = json.load(f)

number_of_generations = len(fitness_results['min_fitness'])
x = [i for i in range(number_of_generations)]

plt.plot(x, fitness_results['min_fitness'], label="Min Fitness")
plt.plot(x, fitness_results['max_fitness'], label="Max Fitness")
plt.plot(x, fitness_results['mean_fitness'], label="Mean Fitness")
plt.legend()
plt.xlabel('Generation number')
plt.ylabel('Fitness')

plt.show()