import numpy as np

def is_convex(polygon):
    def cross_product(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    
    n = len(polygon)
    if n < 3:
        return False
    
    sign = None
    for i in range(n):
        cross_prod = cross_product(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n])
        if cross_prod != 0:
            current_sign = cross_prod > 0
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False
    return True

def get_rectangle():
    print("Enter the two diagonal points of the rectangle:")
    x1, y1 = map(float, input("Enter x1, y1: ").split())
    x2, y2 = map(float, input("Enter x2, y2: ").split())
    return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])


def get_circle():
    try:
        center_x, center_y = map(float, input("Enter center x, y for the circle: ").split())
        radius = float(input("Enter radius of the circle: "))

        if radius <= 0:
            print("Error: Radius must be greater than zero.")
            return None

        theta = np.linspace(0, 2 * np.pi, 100)  # 100 points around the circle
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)

        circle = np.column_stack((x, y))
        return circle

    except ValueError:
        print("Invalid input! Please enter numerical values.")
        return None

def get_ellipse():
    center_x, center_y = map(float, input("Enter center x, y for the ellipse: ").split())
    axis_a = float(input("Enter semi-major axis: "))
    axis_b = float(input("Enter semi-minor axis: "))
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse = np.column_stack((center_x + axis_a * np.cos(theta), center_y + axis_b * np.sin(theta)))
    return ellipse

def is_inside(point, polygon):
    """ Uses ray-casting algorithm to check if a point is inside the polygon """
    x, y = point
    n = len(polygon)
    intersections = 0
    p1x, p1y = polygon[0]
    
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                x_intersect = p1x + (y - p1y) * (p2x - p1x) / (p2y - p1y)
                if x < x_intersect:
                    intersections += 1
        p1x, p1y = p2x, p2y
    
    return intersections % 2 == 1