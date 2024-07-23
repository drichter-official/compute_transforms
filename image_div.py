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
def divide_image(filepath, mini_width, mini_height, output_dir):
    # Read the image
    image = cv2.imread(filepath)
    img_height, img_width, _ = image.shape
    assert img_height % mini_height == 0
    assert img_width % mini_width == 0

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = make_black_pixels_transparent(image, 10)
    # Calculate the number of sub-images horizontally and vertically
    cols = img_width // mini_width
    rows = img_height // mini_height

    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    # Loop through the rows and columns to create sub-images
    for row in tqdm(range(rows)):
        for col in range(cols):
            left = col * mini_width
            upper = row * mini_height
            right = left + mini_width
            lower = upper + mini_height

            # Crop the image using array slicing
            sub_image = cv2.rotate(image[upper:lower, left:right],cv2.ROTATE_180)

            # Create a filename for each sub-image
            sub_image_filename = os.path.join(output_dir, f'sub_image_{base_filename}_{row}_{col}.png')

            # Save the sub-image
            cv2.imwrite(sub_image_filename, sub_image)

    print(f'Successfully saved {rows * cols} sub-images to {output_dir}')



def main():
    input_dir = '/home/daniel/projects/plenoptic_instant_ngp/data/images'
    output_dir = '/home/daniel/projects/plenoptic_transform_instant_ngp/data/image_6x3_transparent_transpose/images'

    prefixs = ['0','1','2','3','4','5']


    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for prefix in prefixs:
        for filename in os.listdir(input_dir):
            if filename.startswith(prefix):
                filepath = os.path.join(input_dir, filename)
                divide_image(filepath, 80, 80, output_dir)
if __name__ == '__main__':
    main()