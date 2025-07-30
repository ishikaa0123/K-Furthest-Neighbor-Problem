import numpy as np
from scipy.spatial.distance import pdist, squareform
from convexpolygon import is_inside
from pso_optimizer import generate_valid_points, ensure_inside
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist

def evaluate(points):
    distances = pdist(points)
    return np.sum(distances ** 2)


def compute_heuristic(candidate_points):
    n = len(candidate_points)
    heuristic = np.zeros(n)
    for i in range(n):
        dist_sum = 0
        for j in range(n):
            if i != j:
                dist_sum += np.linalg.norm(candidate_points[i] - candidate_points[j])
        heuristic[i] = dist_sum / (n - 1)
    heuristic = heuristic / np.max(heuristic)  # normalize
    return heuristic

def ant_colony_optimization(polygon, k, n_ants=50, n_iterations=100, alpha=1, beta=2,
                            evaporation_rate=0.5, q=100):
    from pso_optimizer import generate_valid_points, ensure_inside
    from convexpolygon import is_inside

    candidate_points = generate_valid_points(500, polygon)
    pheromone = np.ones((k, len(candidate_points)))
    heuristic = compute_heuristic(candidate_points)

    best_fitness = -np.inf
    best_solution = None
    fitness_history = []

    for iteration in range(n_iterations):
        solutions = []
        fitness_scores = []

        for ant in range(n_ants):
            solution_indices = []
            for i in range(k):
                prob_numerator = (pheromone[i] ** alpha) * (heuristic ** beta)
                prob_sum = np.sum(prob_numerator)
                if prob_sum == 0 or np.isnan(prob_sum):
                    prob = np.ones(len(candidate_points)) / len(candidate_points)
                else:
                    prob = prob_numerator / prob_sum

                # ε-greedy exploration
                epsilon = 0.1
                if np.random.rand() < epsilon:
                    chosen_idx = np.random.randint(len(candidate_points))
                else:
                    chosen_idx = np.random.choice(len(candidate_points), p=prob)

                solution_indices.append(chosen_idx)

            solution = np.array([candidate_points[idx] for idx in solution_indices])
            solution = ensure_inside(solution, polygon)
            fitness = evaluate(solution)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution

            fitness_scores.append(fitness)
            solutions.append(solution_indices)

        # Pheromone evaporation
        pheromone *= (1 - evaporation_rate)

        # Update with top solutions (elitism)
        sorted_ants = sorted(zip(solutions, fitness_scores), key=lambda x: x[1], reverse=True)
        top_ants = sorted_ants[:5]
        for indices, score in top_ants:
            for i, idx in enumerate(indices):
                pheromone[i][idx] += q * (score / (best_fitness + 1e-6))

        fitness_history.append(best_fitness)

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}/{n_iterations} | Best Fitness: {best_fitness:.4f}")

        # Debug print every 50 iterations
        if (iteration + 1) % 50 == 0:
            print("Pheromone max:", np.max(pheromone))
            print("Pheromone min:", np.min(pheromone))
            print("Sample fitness scores:", fitness_scores[:5])

    return best_solution, best_fitness, fitness_history
