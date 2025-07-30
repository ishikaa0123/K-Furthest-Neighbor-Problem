import numpy as np
from matplotlib.path import Path
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import cProfile
import time

# --- Fast point-in-polygon using Path ---
def fast_is_inside(points, polygon):
    path = Path(polygon)
    return path.contains_points(points)

def fast_is_inside_single(point, polygon):
    path = Path(polygon)
    return path.contains_point(point)

# --- Vectorized valid point generation ---
def generate_valid_points(k, polygon):
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])

    points = []
    while len(points) < k:
        batch = np.column_stack((
            np.random.uniform(min_x, max_x, 5 * k),
            np.random.uniform(min_y, max_y, 5 * k)
        ))
        inside = fast_is_inside(batch, polygon)
        points.extend(batch[inside])
    return np.array(points[:k])

# --- Ensure all points lie inside polygon ---
def ensure_inside(points, polygon):
    inside_mask = fast_is_inside(points, polygon)
    if np.all(inside_mask):
        return points
    else:
        valid = points[inside_mask]
        num_needed = len(points) - len(valid)
        new_points = generate_valid_points(num_needed, polygon)
        return np.vstack((valid, new_points))

# --- Optimized evaluate function ---
# --- Optimized evaluate function using sum of squared distances ---
def evaluate(points):
    distances = pdist(points)
    return np.sum(distances ** 2)


# --- Particle Swarm Optimization with timestamps, elitism, and history tracking ---
def particle_swarm_optimization(polygon, k, num_particles=30, iterations=100, w=0.7, c1=1.5, c2=1.5):
    positions = [generate_valid_points(k, polygon) for _ in range(num_particles)]
    velocities = [np.random.uniform(-1, 1, (k, 2)) for _ in range(num_particles)]
    best_positions = [np.copy(pos) for pos in positions]
    best_fitnesses = [evaluate(pos) for pos in positions]

    global_best_idx = np.argmax(best_fitnesses)
    global_best_position = np.copy(best_positions[global_best_idx])
    global_best_fitness = best_fitnesses[global_best_idx]

    history = []  # Track fitness history over iterations

    print("\n--- Starting PSO ---")
    start_time = time.time()
    for iteration in range(iterations):
        iter_start = time.time()

        # Apply elitism: preserve the best particle
        elite_position = np.copy(global_best_position)
        elite_fitness = global_best_fitness

        for i in range(num_particles):
            r1 = np.random.rand(k, 2)
            r2 = np.random.rand(k, 2)
            cognitive = c1 * r1 * (best_positions[i] - positions[i])
            social = c2 * r2 * (global_best_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            positions[i] += velocities[i]
            positions[i] = ensure_inside(positions[i], polygon)

            fitness = evaluate(positions[i])
            if fitness > best_fitnesses[i]:
                best_fitnesses[i] = fitness
                best_positions[i] = np.copy(positions[i])

                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = np.copy(positions[i])

        # Replace worst particle with elite if needed
        worst_idx = np.argmin(best_fitnesses)
        best_positions[worst_idx] = elite_position
        best_fitnesses[worst_idx] = elite_fitness

        history.append(global_best_fitness)  # Save best fitness for this iteration

        if (iteration + 1) % 100 == 0 or iteration == iterations - 1:
            iter_end = time.time()
            print(f"Iteration {iteration+1}/{iterations} completed in {iter_end - iter_start:.2f}s - Timestamp: {time.strftime('%H:%M:%S')} | Best Fitness: {global_best_fitness:.4f}")

    total_time = time.time() - start_time
    print(f"--- PSO completed in {total_time:.2f} seconds ---\n")
    return global_best_position, global_best_fitness, history

# --- Optional: Profiling toggle ---
def run_with_profiling():
    import main
    cProfile.run('main.main()', sort='time')
