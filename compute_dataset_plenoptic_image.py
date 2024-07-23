import numpy as np
import os, sys
import math
import cv2
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
from scipy.spatial.transform import Rotation as R


def plot_cameras(transform_matrices):
    lim = 0.02#*1e3
    visualizer = CameraPoseVisualizer([-lim, lim], [-lim, lim], [-lim, lim])
    for transform_matrix in transform_matrices:
        if transform_matrix.shape[0] == 3 and transform_matrix.shape[1] == 4:
            transform_matrix = np.vstack([transform_matrix, np.array([0, 0, 0, 1])])
        visualizer.extrinsic2pyramid(transform_matrix, 'c', 2.2e-3)
    visualizer.show()
def calculate_camera_transform(origin, view_direction, up_vector=np.array([0, 1, 0])):
    # Normalize the view direction to get the z-axis of the camera
    z_axis = view_direction / np.linalg.norm(view_direction)
    # Calculate the right (x-axis) vector
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    # Calculate the up (y-axis) vector
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    # Construct the rotation matrix
    rotation_matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], 0],
        [x_axis[1], y_axis[1], z_axis[1], 0],
        [x_axis[2], y_axis[2], z_axis[2], 0],
        [0, 0, 0, 1]
    ])
    # Construct the translation matrix
    translation_matrix = np.array([
        [1, 0, 0, origin[0]],
        [0, 1, 0, origin[1]],
        [0, 0, 1, origin[2]],
        [0, 0, 0, 1]
    ])
    # Combine the rotation and translation matrices to form the transformation matrix
    transform_matrix = np.dot(translation_matrix, rotation_matrix)
    return transform_matrix


def calculate_base_grid(n_img_width, n_img_height, n_focal_x, n_focal_y, pixel_size = 6.5e-3):
    size_x, size_y = pixel_size*n_img_width, pixel_size*n_img_height
    assert n_img_width % n_focal_x == 0
    assert n_img_height % n_focal_y == 0

    pixel_per_focal_x = n_img_width/n_focal_x
    pixel_per_focal_y = n_img_height/n_focal_y
    # Compute Base grid before transform
    # Define the normal vector of the plane
    # z_dir_vec = np.array([0, 0, 1])
    # Define two vectors in the x-y plane
    x_plane_vec = np.array([1, 0, 0])
    y_plane_vec = np.array([0, 1, 0])

    # Create a range of values for the grid
    focal_points_range_x = np.linspace(-size_x/2*(1-pixel_per_focal_x/(2*n_img_width)), size_x/2*(1-pixel_per_focal_x/(2*n_img_width)), n_focal_x)
    focal_points_range_y = np.linspace(-size_y/2*(1-pixel_per_focal_y/(2*n_img_height)), size_y/2*(1-pixel_per_focal_y/(2*n_img_height)), n_focal_y)
    # Create a meshgrid
    u_focal, v_focal = np.meshgrid(focal_points_range_x, focal_points_range_y)
    # Compute the grid points in 3D space
    grid_points_origins = u_focal[:, :, np.newaxis] * x_plane_vec + v_focal[:, :, np.newaxis] * y_plane_vec
    return grid_points_origins


def compute_ray(origin, dir_point):
    # Convert inputs to numpy arrays for vector operations
    focal_point = np.array(dir_point, dtype=float)
    origin = np.array(origin, dtype=float)
    # Calculate the direction vector from the origin to the focal point
    direction = focal_point - origin
    # Normalize the direction vector
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("The focal point and origin cannot be the same point.")
    normalized_direction = direction / norm
    return origin, normalized_direction


def compute_rays(origins, focal_vec, focal_length = 2.2):
    if np.linalg.norm(focal_vec) == 1:
        print("The focalvec is normalized.")
        focal_points = origins+focal_vec*focal_length # needs to be done like this to give focal points the shape of origins
    else:
        print(f"The focalvec is not normalized. {focal_vec}")

        focal_points = origins + focal_vec
    # Compute direction vectors
    direction_vecs = focal_points - origins
    print(direction_vecs.shape)
    # Normalize direction vectors
    norms = np.linalg.norm(direction_vecs, axis=-1, keepdims=True)
    direction_vecs_normalized = direction_vecs / norms
    # Combine origin and direction vectors
    rays = np.stack((origins, direction_vecs_normalized), axis=2)
    return rays

def calculate_transforms(origins, view_directions, up_vector=np.array([0, 1, 0])):
    cols = 27
    rows = 64
    transforms = []

    for row in tqdm(range(rows)):
        for col in range(cols):
            transform = calculate_camera_transform(origins[row][col], view_directions[row][col], up_vector)
            transforms.append(transform)
    return transforms


def calculate_camera_angle(focal_length=2.2*1e-0, sensor_width=1e-3*6.5*80):
    # Calculate the field of view in radians
    fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
    return fov_radians
def transforms_and_imagepaths(base_filename,rows,cols ,origins, view_directions, up_vector=np.array([0, 1, 0]),):
    # Loop through the rows and columns to create sub-images
    image_names = []
    transforms = []
    for row in tqdm(range(rows)):
        for col in range(cols):
            # Create a filename for each sub-image
            image_names.append(f'images/sub_image_{base_filename}_{row}_{col}.png')
            transform = calculate_camera_transform(origins[row][col], view_directions[row][col], up_vector)
            transforms.append(transform)

    return transforms,image_names


def create_json(filename,image_names, transform_matrices, camera_angle):
    frames = []

    for file_name, transform_matrix in zip(image_names, transform_matrices):
        frame = {
            "file_path": file_name,
            "transform_matrix": transform_matrix.tolist()
        }
        frames.append(frame)

    data = {
        "camera_angle_x": camera_angle,
        "frames": frames#,
        #"render_aabb": [
            #[-7, -17, 20],
            #[7, 17, 50]
        #]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def rotate_vector(vector, angle, axis):
    """
    Rotate a 3D vector by a given angle around a specified axis.

    :param vector: Tuple representing the vector (x, y, z)
    :param angle: Angle in degrees to rotate the vector
    :param axis: Axis to rotate around (should be one of 'x', 'y', 'z')
    :return: Rotated vector as a tuple (x', y', z')
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Rotation matrices for each axis
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

    # Apply rotation matrix to the vector
    rotated_vector = np.dot(rotation_matrix, np.array(vector))
    return rotated_vector

def calculate_angle_vectors():
    vecs = []
    for angle in [-34,0,34,-34+60,0+60,34+60,-34+2*60,0+2*60,34+2*60,-34+3*60,0+3*60,34+3*60,-34+4*60,0+4*60,34+4*60,-34+5*60,0+5*60,34+5*60]:
        vecs.append(rotate_vector(np.array([0,0,1]),angle,'y'))
    return vecs

def main():
    n_focal_x = 27
    n_focal_y = 64
    n_img_width = 2160
    n_img_height = 5120

    EL_width = n_img_width / n_focal_x
    EL_height = n_img_height / n_focal_y

    pixel_size = 6.5e-3 * 1e-3
    focal_length = 2.44 * 1e-3

    radius = 40 * 1e-3

    base_names = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2', '4-0', '4-1', '4-2', '5-0', '5-1', '5-2']
    #base_names =    ['0-0']
    print('Generating grid points...')
    grid_points_origins_base = calculate_base_grid(n_img_width,n_img_height, n_focal_x,n_focal_y,pixel_size)
    print(grid_points_origins_base.shape)
    print('Generating origin and view directions for rays...')
    angle_vecs = calculate_angle_vectors()
    transform_list, image_paths_list = [],[]
    for vec,base_name in zip(angle_vecs,base_names):
        target_normal_vector = vec
        target_normal_vector = target_normal_vector / np.linalg.norm(target_normal_vector)  # Normalize

        # Calculate the rotation required to align [0, 0, 1] with the target normal vector
        rotation_vector = np.cross([0, 0, 1], target_normal_vector)
        rotation_angle = np.arccos(np.dot([0, 0, 1], target_normal_vector))
        rotation = R.from_rotvec(rotation_angle * rotation_vector / np.linalg.norm(rotation_vector))

        # Apply the rotation to the grid points
        rotated_grid_points = rotation.apply(grid_points_origins_base.reshape(-1,3)).reshape(n_focal_y,n_focal_x,3)
        rays = compute_rays(rotated_grid_points,target_normal_vector)
        if (target_normal_vector == np.array([0,0,1])).all():
            rays = compute_rays(grid_points_origins_base, target_normal_vector)
        if (target_normal_vector == np.array([0,0,-1])).all():
            rays = compute_rays(grid_points_origins_base, target_normal_vector)

        rays[:, :, 0] = rays[:, :, 0]+vec*radius
        print(rays.shape)
        transforms, image_names = transforms_and_imagepaths(base_name,n_focal_y,n_focal_x, rays[:, :, 0], rays[:, :, 1])
        transform_list.extend(transforms)
        image_paths_list.extend(image_names)
        plot_cameras(transforms)

    camera_angle = calculate_camera_angle(focal_length,pixel_size*80) # in [mm]
    print(camera_angle)

    create_json("data/transforms.json",image_paths_list, transform_list, camera_angle)
    plot_cameras(transform_list)

if __name__ == '__main__':

    main()
