import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="white", font_scale=1.2)

def plot_polygon(polygon, test_points, fitness_history=None):
    plt.figure(figsize=(8, 6))

    polygon_closed = np.vstack([polygon, polygon[0]])
    polygon_edge_color = '#333333'  # Charcoal Black
    polygon_fill_color = '#EEEEEE'  # Light Gray
    
    plt.plot(polygon_closed[:, 0], polygon_closed[:, 1],
             color=polygon_edge_color, linewidth=2.5, label='Polygon')
    plt.fill(polygon_closed[:, 0], polygon_closed[:, 1], 
             color=polygon_fill_color, alpha=0.7)

    if test_points is not None and len(test_points) > 0:
        plt.scatter(test_points[:, 0], test_points[:, 1],
                    color='#800020', s=70, marker='o', edgecolor='black', label='Test Points')

    plt.title("Convex Shape with Optimized Points", fontsize=16, color=polygon_edge_color)
    plt.xlabel("X", fontsize=14, color=polygon_edge_color)
    plt.ylabel("Y", fontsize=14, color=polygon_edge_color)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4, color='#BBBBBB')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    if fitness_history is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(fitness_history, color='#228B22', linewidth=2.5, marker='o')  # Forest Green
        plt.title('GA Fitness Over Generations', fontsize=16, color='#333333')
        plt.xlabel('Generation', fontsize=14, color='#333333')
        plt.ylabel('Fitness Score', fontsize=14, color='#333333')
        plt.grid(True, linestyle='--', alpha=0.4, color='#BBBBBB')
        plt.tight_layout()
        plt.show()
