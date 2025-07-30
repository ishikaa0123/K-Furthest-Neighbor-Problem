import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from convexpolygon import is_convex, get_rectangle, get_circle, get_ellipse, is_inside
from io_operations import get_vertices_from_console, get_vertices_from_csv, get_polygon, get_test_points
from transformations import scale_polygon, rotate_polygon, translate_polygon, shear_polygon
from optimization import fitness_function, select_parents, crossover, mutate, genetic_algorithm
from pso_optimizer import particle_swarm_optimization
from aco_optimizer import ant_colony_optimization
from sa_optimizer import simulated_annealing
from plotting import plot_polygon


def log_to_csv(filename, headers, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


# Add this near the top of your file:
OPTIMIZER_COLORS = {
    "PSO": "#FF6F00",    # Orange
    "GA": "#4CAF50",     # Green
    "ACO": "#2196F3",    # Blue
    "SA": "#9C27B0"      # Purple
}


def plot_fitness_history(history, title, ax, color):
    if history and len(history) > 0:
        x_vals = range(len(history))
        ax.plot(x_vals, history, marker='o', color=color, linestyle='-', label='Fitness')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Fitness", fontsize=14)
        ax.grid(True, linestyle=':', color='gray', linewidth=0.7)
        ax.legend()

        indices_to_annotate = [0, len(history) // 2, len(history) - 1]
        for idx in indices_to_annotate:
            ax.annotate(f"{history[idx]:.2f}",
                        (x_vals[idx], history[idx]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=10,
                        color='black',
                        fontweight='bold')


def plot_polygon_with_points(polygon, points, ax, title, point_color="#039BE5"):
    polygon_edge_color = '#37474F'
    polygon_fill_color = '#E3F2FD'

    ax.plot(np.append(polygon[:, 0], polygon[0, 0]),
            np.append(polygon[:, 1], polygon[0, 1]),
            color=polygon_edge_color, linewidth=2)

    ax.fill(polygon[:, 0], polygon[:, 1], color=polygon_fill_color, alpha=0.4)

    if points is not None and len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1],
                   color=point_color,
                   s=90,
                   edgecolors='black',
                   linewidth=1.2,
                   zorder=5,
                   label='Optimized Points')

    ax.set_title(title, fontsize=16, fontweight='bold', color=polygon_edge_color)
    ax.set_xlabel("X", fontsize=14, color=polygon_edge_color)
    ax.set_ylabel("Y", fontsize=14, color=polygon_edge_color)
    ax.grid(True, linestyle='--', linewidth=0.5, color='#B0BEC5')
    ax.set_aspect('equal')

    if points is not None and len(points) > 0:
       ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)





def safe_int_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(prompt))
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Please enter an integer between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter an integer.")


def safe_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(prompt))
            if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                print(f"Please enter a number between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    

    while True:
        polygon = get_polygon()
        if polygon is None or len(polygon) == 0:
            print("Invalid polygon input. Please enter valid vertices.")
            continue

        plot_polygon(polygon, np.empty((0, 2)))

        while True:
            transform_choice = safe_int_input(
                "Do you want to rotate, translate, scale, or shear the shape? \n"
                " 1. Rotate \n 2. Translate \n 3. Scale \n 4. Shear \n 5. Proceed to test points \nEnter choice: ", 1, 5)

            if transform_choice == 1:
                angle = safe_float_input("Enter the rotation angle in degrees: ")
                polygon = rotate_polygon(polygon, angle)

            elif transform_choice == 2:
                dx, dy = None, None
                while True:
                    try:
                        dx, dy = map(float, input("Enter translation dx and dy separated by space: ").split())
                        break
                    except ValueError:
                        print("Invalid input. Enter two numbers separated by space.")
                polygon = translate_polygon(polygon, dx, dy)

            elif transform_choice == 3:
                scale_factor = safe_float_input("Enter scaling factor: ", min_val=0.0001)
                polygon = scale_polygon(polygon, scale_factor)

            elif transform_choice == 4:
                shear_x, shear_y = None, None
                while True:
                    try:
                        shear_x, shear_y = map(float, input("Enter shear factors for x and y separated by space: ").split())
                        break
                    except ValueError:
                        print("Invalid input. Enter two numbers separated by space.")
                polygon = shear_polygon(polygon, shear_x, shear_y)

            elif transform_choice == 5:
                break

            plot_polygon(polygon, np.empty((0, 2)))

        while True:
            k = safe_int_input("Enter number of k points: ", min_val=1)

            optimizer_choice = safe_int_input(
                "Choose optimizer: \n1. PSO Optimizer \n2. GA Optimizer \n3. ACO Optimizer \n4. SA Optimizer \nEnter choice: ", 1, 4)

            if optimizer_choice == 1:
                iterations = safe_int_input("Enter number of iterations for PSO: ", min_val=1)
                num_particles = safe_int_input("Enter number of particles: ", min_val=1)
                w = safe_float_input("Enter inertia weight (w): ")
                c1 = safe_float_input("Enter cognitive constant (c1): ")
                c2 = safe_float_input("Enter social constant (c2): ")

                analysis_mode = input("Run PSO analysis?\n1. Particle Count Analysis\n2. Iteration Count Analysis\n3. No Analysis\nEnter choice (1-3): ").strip()

                if analysis_mode == "1":
                    particle_range = list(range(30, num_particles + 1, 30))
                    fitness_results = []
                    avg_distances = []
                    csv_data = []

                    print("--- Particle Count Analysis ---")
                    for p_count in particle_range:
                        print(f"Running PSO with {p_count} particles...")
                        best_points, best_fitness, _ = particle_swarm_optimization(
                            polygon, k, num_particles=p_count, iterations=iterations, w=w, c1=c1, c2=c2)
                        fitness_results.append(best_fitness)
                        from scipy.spatial.distance import pdist
                        distances = pdist(best_points)
                        avg_distance = np.mean(distances)
                        avg_distances.append(avg_distance)
                        csv_data.append([p_count, iterations, best_fitness, avg_distance])

                    log_to_csv("particle_count_analysis.csv", ["Particles", "Iterations", "Fitness", "Avg_Pairwise_Distance"], csv_data)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    ax1.plot(particle_range, fitness_results, marker='o', color='green', linewidth=2.5)
                    ax1.set_title('Effect of Particle Count on PSO Fitness')
                    ax1.set_xlabel('Number of Particles')
                    ax1.set_ylabel('Fitness Score')
                    ax1.grid(True, linestyle=':')

                    plot_polygon_with_points(polygon, best_points, ax2, "Convex Shape")

                    plt.tight_layout()
                    plt.show()

                elif analysis_mode == "2":
                    fixed_particles_input = input("Enter particle counts separated by commas (e.g., 200,500): ")
                    try:
                        fixed_particles_list = list(map(int, fixed_particles_input.split(',')))
                    except Exception:
                        print("Invalid input, defaulting to [200, 500]")
                        fixed_particles_list = [200, 500]

                    iteration_list = [500, 1000, 1500, 2000]
                    fitness_result_map = {p: [] for p in fixed_particles_list}
                    csv_data = []

                    print("--- Iteration Count Analysis ---")
                    for p_count in fixed_particles_list:
                        print(f"\nTesting for {p_count} particles...")
                        for iters in iteration_list:
                            print(f"  → Running with {iters} iterations")
                            best_points, best_fitness, _ = particle_swarm_optimization(
                                polygon, k, num_particles=p_count, iterations=iters, w=w, c1=c1, c2=c2
                            )
                            fitness_result_map[p_count].append(best_fitness)
                            csv_data.append([p_count, iters, best_fitness])

                    log_to_csv("iteration_count_analysis.csv", ["Particles", "Iterations", "Fitness"], csv_data)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for p_count in fixed_particles_list:
                        ax.plot(iteration_list, fitness_result_map[p_count], marker='o', linewidth=2, label=f'{p_count} Particles')

                    ax.set_title("Effect of Iterations on PSO Fitness")
                    ax.set_xlabel("Number of Iterations")
                    ax.set_ylabel("Fitness (Higher is better)")
                    ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
                    ax.grid(True, linestyle=':')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()

                print("\nRunning final PSO...")
                best_points, best_fitness, history = particle_swarm_optimization(
                    polygon, k, num_particles=num_particles, iterations=iterations, w=w, c1=c1, c2=c2)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                ax2.set_facecolor('#fff5e6')    
                plot_fitness_history(history, "PSO Fitness Over Iterations", ax1, OPTIMIZER_COLORS["PSO"])
                plot_polygon_with_points(polygon, best_points, ax2, "Convex shape with PSO Optimized Points", OPTIMIZER_COLORS["PSO"])
                
                plt.tight_layout()
                plt.show()

                print(f"Best fitness (sum of distances): {best_fitness}")

            elif optimizer_choice == 2:
                test_points = get_test_points(k, polygon)
                if test_points is None or len(test_points) == 0:
                    print("Invalid test points. Please enter valid points.")
                    continue

                pop_size = safe_int_input("Enter population size: ", min_val=1)
                generations = safe_int_input("Enter number of generations: ", min_val=1)
                crossover_rate = safe_float_input("Enter crossover rate (0-1): ", 0, 1)
                mutation_rate = safe_float_input("Enter mutation rate (0-1): ", 0, 1)

                best_test_points, max_distance, fitness_history = genetic_algorithm(
                    polygon, test_points, pop_size, generations, crossover_rate, mutation_rate)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                ax2.set_facecolor('#f5fff5')    

                plot_fitness_history(fitness_history, "GA Fitness Over Generations", ax1, OPTIMIZER_COLORS["GA"])
                plot_polygon_with_points(polygon, best_test_points, ax2, "Convex shape with GA Optimized Points", OPTIMIZER_COLORS["GA"])

                plt.tight_layout()
                plt.show()

                max_distance, max_pair, distance_matrix = fitness_function(best_test_points)
                print(f"Optimized Maximum Pairwise Distance: {max_distance}")
                print(f"Maximum pairwise distance: {max_distance} between points {max_pair}")
                print("Distance matrix:")
                print(distance_matrix)

            elif optimizer_choice == 3:
                n_ants = safe_int_input("Enter number of ants: ", min_val=1)
                n_iterations = safe_int_input("Enter number of iterations: ", min_val=1)
                alpha = safe_float_input("Enter alpha (pheromone importance): ", 0)
                beta = safe_float_input("Enter beta (heuristic importance): ", 0)
                evaporation = safe_float_input("Enter evaporation rate (0-1): ", 0, 1)
                q = safe_float_input("Enter pheromone constant (Q): ", 0)

                best_points, best_fitness, history = ant_colony_optimization(
                    polygon, k, n_ants, n_iterations, alpha, beta, evaporation,q)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                ax2.set_facecolor('#eaf6ff')    

                plot_fitness_history(history, "ACO Fitness Over Iterations", ax1, OPTIMIZER_COLORS["ACO"])
                plot_polygon_with_points(polygon, best_points, ax2, "Convex shape with ACO Optimized Points", OPTIMIZER_COLORS["ACO"])


                plt.tight_layout()
                plt.show()

                print(f"Best fitness (sum of distances): {best_fitness}")
                log_to_csv("aco_optimizer_results.csv", ["X", "Y"], best_points.tolist())

            elif optimizer_choice == 4:
                initial_temp = safe_float_input("Enter initial temperature: ", min_val=0.0001)
                cooling_rate = safe_float_input("Enter cooling rate (0-1): ", 0, 1)
                iterations = safe_int_input("Enter number of iterations: ", min_val=1)

                best_points, best_fitness, history = simulated_annealing(
                    polygon, k, initial_temp, cooling_rate, iterations)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                ax2.set_facecolor('#f4f0fa')    


                plot_fitness_history(history, "SA Fitness Over Iterations", ax1, OPTIMIZER_COLORS["SA"])
                plot_polygon_with_points(polygon, best_points, ax2, "Convex shape with SA Optimized Points", OPTIMIZER_COLORS["SA"])

                plt.tight_layout()
                plt.show()

                print(f"Best fitness (sum of distances): {best_fitness}")
                log_to_csv("sa_optimizer_results.csv", ["X", "Y"], best_points.tolist())

            else:
                print("Invalid optimizer choice.")
                continue

            rerun = input("Do you want to rerun for the same shape? (yes/no): ").strip().lower()
            if rerun != "yes":
                break

        another_shape = input("Do you want to check another shape? (yes/no): ").strip().lower()
        if another_shape != "yes":
            break


if __name__ == "__main__":
    main()
