# -*- coding: utf-8 -*-
import os
import cv2

# Sort image files by natural order
def natural_sort(images):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(images, key=alphanum_key)

def create_video_from_images(image_folder, output_path, fps=30):
    """Combine images in a specified folder into a video"""
    # Sort images in natural order
    images = natural_sort([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"No images found: {image_folder}")
        return False

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define video encoder and output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v encoder
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to read image: {image_path}")
            continue
        video.write(frame)

    video.release()
    print(f"Video saved to: {output_path}")
    return True

def process_scripted_raw(base_dir, output_dir, fps=30):
    """Process the scripted_raw directory, find all images0 folders and generate videos"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "images6":
                image_folder = os.path.join(root, dir_name)
                relative_path = os.path.relpath(root, base_dir)
                output_video_path = os.path.join(output_dir, f"{relative_path.replace(os.sep, '_')}.mp4")
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

                print(f"Processing folder: {image_folder}")
                create_video_from_images(image_folder, output_video_path, fps=fps)

if __name__ == "__main__":
    base_dir = "E:/code/DataSets/scripted_raw"  # Root directory
    output_dir = "E:/code/DataSets/scripted_videos4"  # Output video directory
    fps = 10  # Video frames per second

    process_scripted_raw(base_dir, output_dir, fps=fps)