import numpy as np
from convexpolygon import is_convex, get_rectangle, get_circle, get_ellipse, is_inside
import csv

def get_vertices_from_console(n):
    vertices = []
    for i in range(n):
        x, y = map(float, input(f"Enter x, y for vertex {i+1}: ").split())
        vertices.append((x, y))
    return np.array(vertices)

def get_vertices_from_csv():
    filename = input("Enter CSV filename: ")
    vertices = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue  # Skip malformed rows
                try:
                    vertices.append((float(row[0]), float(row[1])))
                except ValueError:
                    print(f"Skipping invalid row: {row}")
        return np.array(vertices) if vertices else None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def generate_regular_polygon(n, radius=1.0, center=(0, 0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.array([
        (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
        for angle in angles
    ])

def get_polygon():
    while True:
        choice = int(input("Enter polygon type:\n1. Triangle\n2. Rectangle\n3. Circle\n4. Ellipse\n5. Any other Convex Polygon\nEnter choice (1-5): "))

        if choice == 1:
            return get_vertices_from_console(3)
        elif choice == 2:
            return get_rectangle()
        elif choice == 3:
            return get_circle()
        elif choice == 4:
            return get_ellipse()
        elif choice == 5:
            sub_choice = int(input("Choose input method:\n1. Manual (console/CSV)\n2. Generate regular convex polygon\nEnter choice (1-2): "))
            
            if sub_choice == 1:
                while True:
                    n = int(input("Enter number of vertices for the polygon: "))
                    method = int(input("Choose input method for polygon vertices:\n1. Console\n2. CSV File\nEnter choice (1-2): "))

                    if method == 1:
                        polygon = get_vertices_from_console(n)
                    elif method == 2:
                        polygon = get_vertices_from_csv()
                        if polygon is None:
                            print("Error reading CSV file. Please enter vertices manually.")
                            polygon = get_vertices_from_console(n)
                    else:
                        print("Invalid choice. Try again.")
                        continue

                    if is_convex(polygon):
                        return polygon
                    else:
                        print("The polygon is not convex. Please enter a convex polygon.")
            elif sub_choice == 2:
                n = int(input("Enter number of vertices for the regular polygon (>=3): "))
                if n < 3:
                    print("A polygon must have at least 3 sides.")
                    continue
                radius = float(input("Enter radius of the regular polygon: "))
                return generate_regular_polygon(n, radius)
            else:
                print("Invalid sub-choice. Try again.")

def get_test_points(k, polygon):
    method = int(input("Choose input method for test points:\n1. Console\n2. CSV File\n3. Generate randomly\nEnter choice (1-3): "))

    points = []

    if method == 1:
        for i in range(k):
            while True:
                x, y = map(float, input(f"Enter x, y for test point {i+1}: ").split())
                if is_inside((x, y), polygon):
                    points.append((x, y))
                    break
                else:
                    print("Point is outside the polygon. Please enter a valid point.")

    elif method == 2:
        points = get_vertices_from_csv()
        if points is None:
            print("Error reading CSV file. Please enter points manually.")
            return get_test_points(k, polygon)
        points = [p for p in points if is_inside(p, polygon)]
        while len(points) < k:
            print(f"Only {len(points)} valid points found. Please enter {k - len(points)} more points.")
            points.extend(get_test_points(k - len(points), polygon))

    else:
        min_x, max_x = min(polygon[:, 0]), max(polygon[:, 0])
        min_y, max_y = min(polygon[:, 1]), max(polygon[:, 1])

        while len(points) < k:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if is_inside((x, y), polygon):
                points.append((x, y))

    return np.array(points)
