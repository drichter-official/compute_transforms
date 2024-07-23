import numpy as np

def rotate_vector(vector, angle, axis):
    angle_rad = np.deg2rad(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                    [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                                    [0, 1, 0],
                                    [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', 'z'.")
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

def calculate_angle_vectors(angles, axis='y'):
    vecs = []
    for angle in angles:
        vecs.append(rotate_vector(np.array([0, 0, 1]), angle, axis))
    return vecs

# Example usage
angles = [0, 90, 180, 270]
axis = 'y'
rotated_vectors = calculate_angle_vectors(angles, axis)
print(rotated_vectors)