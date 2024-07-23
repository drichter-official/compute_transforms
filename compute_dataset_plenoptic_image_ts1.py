import numpy as np
import os, sys
import math
import cv2
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from util.camera_pose_visualizer import CameraPoseVisualizer

def plot_cameras(transform_matrices):
    visualizer = CameraPoseVisualizer([-15, 15], [-15, 15], [-30, 5])
    for transform_matrix in transform_matrices:
        if transform_matrix.shape[0] == 3 and transform_matrix.shape[1] == 4:
            transform_matrix = np.vstack([transform_matrix, np.array([0, 0, 0, 1])])
        visualizer.extrinsic2pyramid(transform_matrix, 'c', 1)
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


def calculate_base_grid(n_img_width, n_img_height, n_focal_x, n_focal_y, pixel_size = 6.5e-3, focal_length = 2.2*1e-0):
    size_x, size_y = pixel_size*n_img_width, pixel_size*n_img_height
    #size_x, size_y =n_img_x, n_img_y
    z_off_set = 0 #27.5*1e-0
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
    grid_points_focal = u_focal[:, :, np.newaxis] * x_plane_vec + v_focal[:, :, np.newaxis] * y_plane_vec

    return grid_points_focal-np.array([0,0,1])*z_off_set,grid_points_focal-np.array([0,0,1])*z_off_set+np.array([0,0,1])*focal_length


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


def compute_rays(origins, focal_points):
    # Ensure inputs are numpy arrays
    origins = np.array(origins, dtype=np.float32)
    focal_points = np.array(focal_points, dtype=np.float32)
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
def transforms_and_imagepaths(filepath, mini_width, mini_height ,origins, view_directions, up_vector=np.array([0, 1, 0]),):
    image = cv2.imread(filepath)
    img_height, img_width, _ = image.shape
    assert img_height % mini_height == 0
    assert img_width % mini_width == 0

    # Calculate the number of sub-images horizontally and vertically
    cols = img_width // mini_width
    rows = img_height // mini_height

    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    base_filename ='0-0'
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


def plot_figure(plenoptic_rays_flat,grid_points_flat_image,grid_points_flat_focal):
    # Plot the original and rotated grid points
    fig = plt.figure(figsize=(14, 7))

    # Plot original grid
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(
        0, 0, 0, 0, 0, 1,
        length=1, color='r'
    )
    for i in range(len(plenoptic_rays_flat)):
        ax1.quiver(plenoptic_rays_flat[i][0][0],plenoptic_rays_flat[i][0][1],plenoptic_rays_flat[i][0][2],plenoptic_rays_flat[i][1][0],plenoptic_rays_flat[i][1][1],plenoptic_rays_flat[i][1][2],length=1, color='r')
    ax1.scatter(grid_points_flat_image[:, 0], grid_points_flat_image[:, 1], grid_points_flat_image[:, 2], s=1)
    ax1.scatter(grid_points_flat_focal[:, 0], grid_points_flat_focal[:, 1], grid_points_flat_focal[:, 2], s=1)

    ax1.set_title('Original Grid')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()

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

def main():

    print('Generating grid points...')
    grid_points_origins_base, grid_points_focal_base = calculate_base_grid( 2160,5120, 27,64)
    print(grid_points_origins_base.shape)
    print('Generating origin and view directions for rays...')
    rays = compute_rays(grid_points_origins_base,grid_points_focal_base)
    print(rays.shape)
    #plot_figure(rays.reshape(-1,2,3), grid_points_origins.reshape(-1,3),grid_points_focal.reshape(-1,3))

    rays_flat = rays.reshape(-1,2,3)
    transforms = calculate_transforms(rays[:,:,0], rays[:,:,1])
    camera_angle = calculate_camera_angle() # in [mm]
    print(transforms)
    print(camera_angle)

    transforms, image_names = transforms_and_imagepaths("/home/daniel/projects/plenoptic_transform_instant_ngp/1.png", 80, 80, rays[:,:,0], rays[:,:,1])
    print(image_names)
    create_json("data/transforms.json",image_names, transforms, camera_angle)
    plot_cameras(transforms)

if __name__ == '__main__':

    main()
