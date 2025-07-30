import numpy as np

def scale_polygon(polygon, scale_factor):
    centroid = np.mean(polygon, axis=0)
    scaled_polygon = centroid + scale_factor * (polygon - centroid)
    return np.round(scaled_polygon, decimals=6)

def rotate_polygon(polygon, angle):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    centroid = np.mean(polygon, axis=0)
    rotated_polygon = np.dot(polygon - centroid, rotation_matrix.T) + centroid
    
    return np.round(rotated_polygon, decimals=6)

def translate_polygon(polygon, tx, ty):
    translated_polygon = polygon + np.array([tx, ty])
    return np.round(translated_polygon, decimals=6)

def shear_polygon(polygon, shear_x, shear_y):
    shear_matrix = np.array([
        [1, shear_x],
        [shear_y, 1]
    ])
    
    centroid = np.mean(polygon, axis=0)
    sheared_polygon = np.dot(polygon - centroid, shear_matrix.T) + centroid
    
    return np.round(sheared_polygon, decimals=6)
