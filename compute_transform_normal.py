import numpy as np
import os, sys
import math
import cv2
from tqdm import tqdm
import json

import argparse
import configparser

def calculate_camera_transform(origin, view_direction, up_vector=np.array([0, 0, 1])):
    z_axis = view_direction / np.linalg.norm(view_direction)
    x_axis = np.cross(up_vector, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], 0],
        [x_axis[1], y_axis[1], z_axis[1], 0],
        [x_axis[2], y_axis[2], z_axis[2], 0],
        [0, 0, 0, 1]
    ])
    translation_matrix = np.array([
        [1, 0, 0, origin[0]],
        [0, 1, 0, origin[1]],
        [0, 0, 1, origin[2]],
        [0, 0, 0, 1]
    ])
    transform_matrix = np.dot(translation_matrix, rotation_matrix)
    return transform_matrix

def compute_rays(origins, focal_vec, focal_length=2.2):
    if np.linalg.norm(focal_vec) == 1:
        focal_points = origins + focal_vec * focal_length
    else:
        focal_points = origins + focal_vec
    direction_vecs = focal_points - origins
    norms = np.linalg.norm(direction_vecs, axis=-1, keepdims=True)
    direction_vecs_normalized = direction_vecs / norms
    rays = np.stack((origins, direction_vecs_normalized), axis=2)
    return rays

def calculate_camera_angle(focal_length=2.44, sensor_width=6.5 * 1e-3 * 80):
    fov_radians = 2 * math.atan(sensor_width / (2 * focal_length))
    return fov_radians

def create_json(filename, image_names, transform_matrices, camera_angle_x):
    frames = []
    for file_name, transform_matrix in zip(image_names, transform_matrices):
        frame = {
            "file_path": file_name,
            "transform_matrix": transform_matrix.tolist()
        }
        frames.append(frame)
    data = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def generate_positions(radius, center, n_pos, axis='Y', random=False, type='CIRCLE'):
    if type == 'CIRCLE':
        if random:
            angles = np.random.uniform(0, 2 * np.pi, n_pos)
        else:
            angles = np.linspace(0, 2 * np.pi, n_pos, endpoint=False)

        if axis == 'X':
            positions = np.column_stack((np.zeros(n_pos), radius * np.cos(angles), radius * np.sin(angles)))
        elif axis == 'Y':
            positions = np.column_stack((radius * np.cos(angles), np.zeros(n_pos), radius * np.sin(angles)))
        elif axis == 'Z':
            positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(n_pos)))
        else:
            raise ValueError("Invalid axis. Choose from 'X', 'Y', or 'Z'.")
    elif type == 'SPHERE':
        phi = np.random.uniform(0, 2 * np.pi, n_pos)
        if random:
            theta = np.arccos(2 * np.random.uniform(0, 1, n_pos) - 1)
        else:
            indices = np.arange(0, n_pos, dtype=float) + 0.5
            theta = np.arccos(1 - 2 * indices / n_pos)

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        positions = np.column_stack((x, y, z))+np.array(center)
    else:
        raise ValueError("Invalid type. Choose from 'CIRCLE' or 'SPHERE'.")

    return positions
def compute_vectors_to_origin(positions, origin):
    positions = np.array(positions)
    origin = np.array(origin)
    vectors = origin - positions
    return vectors


def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config['DEFAULT']

def main():
    parser = argparse.ArgumentParser(description='Process some paths and parameters.')
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--sampling', type=str, default='CIRCLE')
    parser.add_argument('--num_images', type=int, default=60)
    parser.add_argument('--random_sampling', type=bool, default=False, help='normalize for instant_ngp box')

    parser.add_argument('--radius', type=float, default=4)
    parser.add_argument('--center', type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument('--rot_axis', type=str, default='Y')

    parser.add_argument('--camera_up',type=float, nargs=3, default=[0, 0, 1])
    parser.add_argument('--focal_length', type=float, default=40)  # Default normal camera focal length
    parser.add_argument('--sensor_size', type=float, default=36)  # Default normal camera sensor width

    parser.add_argument('--normalize', type=bool, default=False, help='normalize for instant_ngp box')
    parser.add_argument('--visualize', action='store_true', help='visualize in image saved in output directory')

    parser.add_argument('--config_path', type=str, default='./configs/config_normal.ini', help='Path to load the configuration file')
    parser.add_argument('--load_config', action='store_true', default=True, help='Load configuration from file')
    args = parser.parse_args()

    if args.load_config:
        print('Loading configuration from file')
        config = load_config(args.config_path)
        args.output_dir = config.get('output_dir', args.output_dir)
        args.sampling = config.get('sampling', args.sampling)
        args.num_images = int(config.get('num_images', args.num_images))
        args.random_sampling = config.getboolean('random_sampling', args.random_sampling)

        args.radius = float(config.get('radius', args.radius))
        args.center = list(map(float, config.get('center', ','.join(map(str, args.center))).split(',')))
        args.rot_axis = config.get('rot_axis', args.rot_axis)

        args.camera_up = list(map(float, config.get('camera_up', ','.join(map(str, args.camera_up))).split(',')))
        args.focal_length = float(config.get('focal_length', args.focal_length))
        args.sensor_size = float(config.get('sensor_size', args.sensor_size))

        args.normalize = config.getboolean('normalize', args.normalize)
        args.visualize = config.getboolean('visualize', args.visualize)

    print(
        f"radius = {args.radius}, "
        f"number images = {args.num_images}, "
        f"rotation axis = {args.rot_axis}, "
        f"random = {args.random_sampling}, "
        f"sampling = {args.sampling}"
    )

    camera_positions = generate_positions(args.radius,args.center, args.num_images, args.rot_axis, args.random_sampling, args.sampling)
    view_dirs = compute_vectors_to_origin(camera_positions, args.center)

    transform_list, image_paths_list = [], []
    print('Computing transform matrices')
    for idx, (pos,vec) in enumerate(zip(camera_positions,view_dirs)):
        view_direction = vec / np.linalg.norm(vec)
        transform_matrix = calculate_camera_transform(pos, -view_direction, np.array(args.camera_up))
        transform_list.append(transform_matrix)
        image_paths_list.append('images/' + str(idx) + '.png')

    camera_angle_x = calculate_camera_angle(args.focal_length, args.sensor_size)
    print('Saving transform file')
    create_json(args.output_dir+"/transforms.json", image_paths_list, transform_list, camera_angle_x)

if __name__ == "__main__":
    main()
