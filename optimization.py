import numpy as np
import random
from convexpolygon import is_inside
from scipy.spatial.distance import pdist, squareform

def fitness_function(points):
    pairwise_distances = pdist(points)
    if len(pairwise_distances) == 0:
        return 0, None, None

    fitness = np.sum(pairwise_distances ** 2)
    dist_matrix = squareform(pairwise_distances)

    return fitness, None, dist_matrix



# --- Parent Selection: Tournament or Weighted Random ---
def select_parents(population, fitness_scores):
    selected = random.choices(population, weights=fitness_scores, k=2)
    return selected[0], selected[1]

# --- Corrected Uniform Crossover ---
def crossover(parent1, parent2, polygon):
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    for i in range(len(parent1)):
        if random.random() < 0.5:  # uniform crossover
            child1[i], child2[i] = parent2[i], parent1[i]
    # Ensure points inside polygon
    child1 = np.array([p if is_inside(p, polygon) else get_random_point_in_polygon(polygon) for p in child1])
    child2 = np.array([p if is_inside(p, polygon) else get_random_point_in_polygon(polygon) for p in child2])
    return child1, child2

# --- Mutation ---
def mutate(child, polygon, mutation_rate=0.1):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            dx, dy = np.random.uniform(-1, 1, size=2)
            new_point = child[i] + [dx, dy]
            if is_inside(new_point, polygon):
                child[i] = new_point
            else:
                child[i] = get_random_point_in_polygon(polygon)
    return child

# --- Main Genetic Algorithm ---
def genetic_algorithm(polygon, test_points, pop_size, generations, mutation_rate, crossover_rate):
    population = [np.copy(test_points) for _ in range(pop_size)]
    best_solution = None
    best_fitness = -np.inf
    fitness_history = []

    for generation in range(generations):
        fitness_scores = [fitness_function(ind)[0] for ind in population]

        current_best_fitness = max(fitness_scores)
        fitness_history.append(current_best_fitness)

        current_best_idx = np.argmax(fitness_scores)
        current_best = population[current_best_idx]
        if current_best_fitness > best_fitness:
            best_solution = current_best
            best_fitness = current_best_fitness

        new_population = [best_solution]  # Elitism

        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitness_scores)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, polygon)
            else:
                child1, child2 = np.copy(parent1), np.copy(parent2)

            child1 = mutate(child1, polygon, mutation_rate)
            child2 = mutate(child2, polygon, mutation_rate)

            child1 = ensure_valid(child1, test_points, polygon)
            child2 = ensure_valid(child2, test_points, polygon)

            new_population.extend([child1, child2])

        population = new_population[:pop_size]

    return best_solution, float(best_fitness), fitness_history

# --- Ensure All Points Are Valid ---
def ensure_valid(child, reference, polygon):
    child = np.unique(child, axis=0)
    while len(child) < len(reference):
        child = np.vstack((child, get_random_point_in_polygon(polygon)))
    return child

# --- Generate Random Valid Point Inside Polygon ---
def get_random_point_in_polygon(polygon):
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    while True:
        x_rand = np.random.uniform(x_min, x_max)
        y_rand = np.random.uniform(y_min, y_max)
        if is_inside((x_rand, y_rand), polygon):
            return np.array([x_rand, y_rand])
