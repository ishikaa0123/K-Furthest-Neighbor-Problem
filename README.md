# K-Furthest-Neighbor-Problem
This repository presents my M.Sc. dissertation project titled "A Performance-Based Study of Different Optimization Techniques on the k-Furthest Neighbor Problem."

Problem Statement:

 The objective is to optimally place k points within a bounded region such that the sum of squared pairwise distances among the points is maximized.


In this work, I implemented and evaluated four metaheuristic optimization algorithms that are Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), and Simulated Annealing (SA), to solve the k-Furthest Neighbor Problem only on the convex shapes for now using python.

Libraries Used: numpy, scipy, matplotlib

The performance of these algorithms was compared across three different values of k (3, 4, 5)and five iteration settings: 200, 500, 1000, 2000, and 5000 over 5 convex shapes (Circle, , rectangle, triangle, ellipse and hexagon or any other convex polygon) with any orientation of the shape.

The project includes clean and modular Python code, visualizations of the point placements, and detailed performance analysis to highlight the effectiveness and convergence behavior of each algorithm.
