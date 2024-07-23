import numpy as np
import os, sys
import math
import cv2
from tqdm import tqdm
import json

import argparse
import configparser

def calculate_camera_transform(origin, view_direction, up_vector=np.array([0, 1, 0])):
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

def create_json(filename, image_names, transform_matrices, camera_angle_x, camera_angle_y):
    frames = []
    for file_name, transform_matrix in zip(image_names, transform_matrices):
        frame = {
            "file_path": file_name,
            "transform_matrix": transform_matrix.tolist()
        }
        frames.append(frame)
    data = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "frames": frames
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

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
    rotated_vector = np.dot(rotation_matrix, np.array(vector))
    return rotated_vector

def calculate_angle_vectors(angles, axis='y'):
    vecs = []
    for angle in angles:
        vecs.append(rotate_vector(np.array([0, 0, 1]), angle, axis))
    return vecs

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config['DEFAULT']

def main():
    parser = argparse.ArgumentParser(description='Process some paths and parameters.')
    parser.add_argument('--output_dir', type=str, default='/home/daniel/projects/plenoptic_transform_instant_ngp/data')
    parser.add_argument('--sampling', type=str, default='CIRCLE')
    parser.add_argument('--num_images', type=int, default=60)
    parser.add_argument('--radius', type=float, default=4)
    parser.add_argument('--center', type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument('--rot_axis', type=str, nargs=1, default='Y')

    parser.add_argument('--focal_length', type=float, default=40)  # Default normal camera focal length
    parser.add_argument('--sensor_size', type=float,nargs=1, default=36)  # Default normal camera sensor width

    parser.add_argument('--normalize', action='store_true',default='True', help='normalize for instant_ngp box')
    parser.add_argument('--visualize', action='store_true',default='False', help='visualize in image saved in output directory')

    parser.add_argument('--config_path', type=str, default='/configs/config_normal.ini', help='Path to load the configuration file')
    parser.add_argument('--load_config', action='store_true',default='True', help='Load configuration from file')
    args = parser.parse_args()

    if args.load_config:
        config = load_config(args.config_path)
        args.output_dir = config.get('output_dir', args.output_dir)
        args.sampling = config.get('sampling', args.sampling)
        args.num_images = int(config.get('num_images', args.num_images))
        args.center = list(map(float, config.get('center', ','.join(map(str, args.center))).split(',')))
        args.camera_rotation = list(map(float, config.get('camera_rotation', ','.join(map(str, args.camera_rotation))).split(',')))
        args.rot_axis = config.get('rot_axis', args.rot_axis)

        args.focal_length = float(config.get('focal_length', args.focal_length))
        args.sensor_size = float(config.get('sensor_size', args.sensor_size))

        args.normalize = bool(config.get('normalize', args.normalize))


    #camera_angles = [-34, 0, 34, 26, 60, 94, 86, 120, 154, 146, 180, 214, 206, 240, 274, 266, 300, 334]

    if len(camera_angles) == 0:
        camera_angles = np.arange(0, 360, 360 / args.num_images)
    print('Generating origin and view directions for rays...')
    angle_vecs = calculate_angle_vectors(camera_angles)
    transform_list, image_paths_list = [], []
    for idx, vec in enumerate(angle_vecs):
        view_direction = vec / np.linalg.norm(vec)
        transform_matrix = calculate_camera_transform(vec * cam_radius, view_direction)
        transform_list.append(transform_matrix)
        image_paths_list.append('images/' + str(idx) + '.png')

    camera_angle_x = calculate_camera_angle(focal_length, sensor_width[0])
    camera_angle_y = calculate_camera_angle(focal_length, sensor_width[1])

    create_json("data/transforms.json", image_paths_list, transform_list, camera_angle_x, camera_angle_y)

if __name__ == "__main__":
    main()
