import os
from io import BytesIO
import h5py
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Root directory path
root_path = "/home/ubuntu/Desktop/dataset/h5_ur_1rgb"
output_dir = "output_videos"  # Directory for output videos

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Video writer parameters
fps = 30  # Frame rate
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Traverse all subdirectories under the root directory
for root, dirs, files in os.walk(root_path):
    # Check if "success_episodes" is in the current path
    if "success_episodes" in root:
        for file_name in files:
            if file_name.endswith(".hdf5"):
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")

                # Generate a unique output video path
                relative_path = os.path.relpath(root, root_path)
                unique_name = os.path.join(relative_path, file_name.replace(".hdf5", ".mp4"))
                unique_name = unique_name.replace(os.sep, "_")  # Replace path separators to avoid invalid filenames
                output_video_path = os.path.join(output_dir, unique_name)

                # Process the HDF5 file
                with h5py.File(file_path, "r") as hdf_file:
                    rgb_images = hdf_file["observations"]["rgb_images"]["camera_top"]
                    frame_count = len(rgb_images)

                    # Get the dimensions of the first frame
                    if frame_count > 0:
                        first_image = Image.open(BytesIO(rgb_images[0]))
                        frame_width, frame_height = first_image.size

                        # Initialize the video writer
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Could not open video writer for: {output_video_path}")

                        # Decode frame by frame and write to the video
                        for i in tqdm(range(frame_count), desc=f"Processing {file_name}"):
                            image = Image.open(BytesIO(rgb_images[i]))
                            image_np = np.array(image)
                            frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame_bgr)

                        # Release the current video writer
                        video_writer.release()
                        print(f"Video generated successfully: {output_video_path}")
                    else:
                        print(f"Skipping empty HDF5 file: {file_path}")


print(f"All videos have been generated and saved in the directory: {output_dir}")