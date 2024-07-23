import numpy as np
import os
import cv2
from tqdm import tqdm

def make_black_pixels_transparent(image, threshold=10):
    """Convert black or near-black pixels to transparent in an image."""
    # Convert the image to RGBA
    if image.shape[2] == 3:  # If image is RGB, convert to RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create a mask of pixels that are near black
    black_pixels = np.all(image[:, :, :3] < threshold, axis=-1)

    # Set the alpha channel to 0 for these pixels
    image[black_pixels, 3] = 0

    return image

def concatenate_images(sub_images_dir, output_filepath, base_names ,mini_width, mini_height, rows, cols):
    # Create an empty canvas for the full image
    full_height = rows * mini_height
    full_width = cols * mini_width
    full_image = np.zeros((full_height, full_width, 4), dtype=np.uint8)

    # Loop through the sub-images and place them into the full image
    for base_name in base_names:
        for row in tqdm(range(rows)):
            for col in range(cols):
                sub_image_filename = os.path.join(sub_images_dir, f'sub_image_{base_name}_{row}_{col}.png')
                if not os.path.exists(sub_image_filename):
                    print(f"Warning: {sub_image_filename} does not exist.")
                    continue
                sub_image = cv2.imread(sub_image_filename, cv2.IMREAD_UNCHANGED)
                if sub_image is None:
                    print(f"Warning: Failed to read {sub_image_filename}. It might be corrupted or not a valid image.")
                    continue
                sub_image = make_black_pixels_transparent(sub_image, 10)
                sub_image = cv2.rotate(sub_image, cv2.ROTATE_180)
                upper = row * mini_height
                left = col * mini_width
                full_image[upper:upper + mini_height, left:left + mini_width] = sub_image

        # Save the full image
        cv2.imwrite(output_filepath+f'/{base_name}.png', full_image)
        print(f'Successfully saved the full image to {output_filepath}')

def main():
    sub_images_dir = '/home/daniel/projects/plenoptic_transform_instant_ngp/data/bsim_3x6_rm_sc3/images'
    output_filepath = '/home/daniel/projects/plenoptic_transform_instant_ngp/data/bsim_3x6_rm_sc3/image_full'
    mini_width = 80
    mini_height = 80
    rows = 64
    cols = 27
    base_names = ['0','1','2','3','4','5']

    concatenate_images(sub_images_dir, output_filepath, base_names, mini_width, mini_height, rows, cols)

if __name__ == '__main__':
    main()