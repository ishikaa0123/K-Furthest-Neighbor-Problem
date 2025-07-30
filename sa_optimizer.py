import numpy as np
import random
from matplotlib.path import Path

def calculate_total_distance(points):
    dist_sum = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist_sum += np.sum((points[i] - points[j]) ** 2)  # Squared distance
    return dist_sum

def point_in_polygon(point, polygon):
    return Path(polygon).contains_point(point)

def generate_random_points_in_polygon(polygon, k):
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    points = []
    while len(points) < k:
        p = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if point_in_polygon(p, polygon):
            points.append(p)
    return np.array(points)

def simulated_annealing(polygon, k, initial_temp=1.0, cooling_rate=0.995, iterations=2000):
    current_points = generate_random_points_in_polygon(polygon, k)
    current_fitness = calculate_total_distance(current_points)
    best_points = current_points.copy()
    best_fitness = current_fitness
    fitness_history = [best_fitness]

    temp = initial_temp

    for i in range(iterations):
        new_points = current_points + np.random.normal(0, 0.01, current_points.shape)
        for idx, point in enumerate(new_points):
            if not point_in_polygon(point, polygon):
                new_points[idx] = current_points[idx]

        new_fitness = calculate_total_distance(new_points)

        if new_fitness > current_fitness or random.random() < np.exp((new_fitness - current_fitness) / temp):
            current_points = new_points
            current_fitness = new_fitness

            if new_fitness > best_fitness:
                best_points = new_points.copy()
                best_fitness = new_fitness

        fitness_history.append(best_fitness)
        temp *= cooling_rate

    return best_points, best_fitness, fitness_history
