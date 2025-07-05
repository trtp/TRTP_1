import os
import random
import shutil


def random_sample_images(source_folder, output_folder, num_samples):
    """
    Randomly samples a specified number of images from a source folder
    and saves them to a target folder.

    :param source_folder: The path to the source image folder.
    :param output_folder: The path to the target folder where sampled images will be saved.
    :param num_samples: The number of images to sample.
    """
    # Get the list of image files
    image_files = [f for f in os.listdir(source_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if len(image_files) < num_samples:
        print(f"Not enough images to sample. The source folder only contains {len(image_files)} images.")
        return

    # Randomly sample the images
    selected_images = random.sample(image_files, num_samples)

    # Create the target folder (if it doesn't exist)
    os.makedirs(output_folder, exist_ok=True)

    # Copy the selected images to the target folder
    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        target_path = os.path.join(output_folder, image)
        shutil.copy(source_path, target_path)
        print(f"Sampled and copied: {image} -> {target_path}")

    print(f"Random sampling complete. A total of {num_samples} images were saved to {output_folder}")


# Example usage
source_folder = "/home/ubuntu/Desktop/dataset/droidCutImage"  # Replace with your source image folder path
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # Replace with your target folder path
num_samples = 1000  # Replace with the number of images to sample

random_sample_images(source_folder, output_folder, num_samples)