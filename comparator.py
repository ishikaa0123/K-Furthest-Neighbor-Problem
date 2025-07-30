import matplotlib.pyplot as plt
import numpy as np
from pso_optimizer import particle_swarm_optimization
from aco_optimizer import ant_colony_optimization
from sa_optimizer import simulated_annealing
from optimization import genetic_algorithm, fitness_function
from io_operations import get_polygon, get_test_points

# Define parameter grid
k_values = [3, 4, 5]
iteration_values = [200, 500, 1000, 2000, 5000]

# Fixed parameters for each algorithm
NUM_PARTICLES = 200
POP_SIZE = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.001
N_ANTS = 200
ALPHA = 1.2
BETA = 1.2
EVAPORATION = 0.6
Q = 100
INIT_TEMP = 6000
COOLING_RATE = 0.95

OPTIMIZER_COLORS = {
    "PSO": "#FF6F00",
    "GA": "#4CAF50",
    "ACO": "#2196F3",
    "SA": "#9C27B0"
}

# Get polygon once
polygon = get_polygon()
if polygon is None or len(polygon) == 0:
    print("Invalid polygon input.")
    exit()

# Run for each k and iteration
for k in k_values:
    for iterations in iteration_values:
        print(f"\n========== Running for k={k}, Iterations={iterations} ==========")
        results = {}

        # PSO
        pso_points, pso_fitness, pso_history = particle_swarm_optimization(
            polygon, k, iterations=iterations, num_particles=NUM_PARTICLES, w=0.5, c1=1.5, c2=2.0)
        results["PSO"] = {"fitness": pso_fitness, "history": pso_history}

        # GA
        test_points = get_test_points(k, polygon)
        ga_points, _, ga_history = genetic_algorithm(
            polygon, test_points, pop_size=POP_SIZE, generations=iterations,
            crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE)
        ga_fitness, _, _ = fitness_function(ga_points)
        results["GA"] = {"fitness": ga_fitness, "history": ga_history}

        # ACO
        aco_points, aco_fitness, aco_history = ant_colony_optimization(
            polygon, k, n_ants=N_ANTS, n_iterations=iterations, alpha=ALPHA,
            beta=BETA, evaporation_rate=EVAPORATION, q=Q)
        results["ACO"] = {"fitness": aco_fitness, "history": aco_history}

        # SA
        sa_points, sa_fitness, sa_history = simulated_annealing(
            polygon, k, initial_temp=INIT_TEMP, cooling_rate=COOLING_RATE, iterations=iterations)
        results["SA"] = {"fitness": sa_fitness, "history": sa_history}

        # --- Plot: Bar chart ---
        plt.figure(figsize=(10, 5))
        optimizers = list(results.keys())
        fitness_values = [results[o]["fitness"] for o in optimizers]
        colors = [OPTIMIZER_COLORS[o] for o in optimizers]

        plt.bar(optimizers, fitness_values, color=colors)
        plt.title(f"Final Fitness Comparison (k={k}, iter={iterations})", fontsize=16, fontweight='bold')
        plt.xlabel("Optimizer", fontsize=14)
        plt.ylabel("Fitness (Sum of Distances)", fontsize=14)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"final_fitness_k{k}_iter{iterations}.png", dpi=300)
        plt.close()

        # --- Plot: Line plot ---
        plt.figure(figsize=(12, 6))
        for opt_name, data in results.items():
            history = data["history"]
            if history:
                plt.plot(history, label=opt_name, color=OPTIMIZER_COLORS[opt_name], linewidth=2)
        plt.title(f"Fitness Over Iterations (k={k}, iter={iterations})", fontsize=16, fontweight='bold')
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Fitness", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.savefig(f"fitness_over_iterations_k{k}_iter{iterations}.png", dpi=300)
        plt.close()
