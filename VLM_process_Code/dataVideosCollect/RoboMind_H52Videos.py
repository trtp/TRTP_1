from io import BytesIO
import h5py
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Open the HDF5 file
file_path = "/home/ubuntu/Desktop/dataset/h5_ur_1rgb/bread_in_basket_1/success_episodes/train/1014_140258/data/trajectory.hdf5"
output_video_path = "../../output_video.mp4"

with h5py.File(file_path, "r") as hdf_file:
    # Access the RGB image data
    rgb_images = hdf_file["observations"]["rgb_images"]["camera_top"]
    frame_count = len(rgb_images)

    # Decode the first frame to get its dimensions
    first_image = Image.open(BytesIO(rgb_images[0]))
    frame_width, frame_height = first_image.size

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Using an H.264 encoder may require installing appropriate codecs
    fps = 30  # Assuming a frame rate of 30, adjust as needed
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not video_writer.isOpened():
        raise RuntimeError("Video writer could not be opened. Check the codec parameters and output path!")

    # Decode and write each frame
    for i in tqdm(range(frame_count), desc="Processing frames"):
        # Decode the byte stream into an image
        image = Image.open(BytesIO(rgb_images[i]))
        image_np = np.array(image)  # Convert to a NumPy array
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR format

        # Check if the frame dimensions are correct
        if frame_bgr.shape[:2] != (frame_height, frame_width):
            raise ValueError(f"Frame {i} has mismatched dimensions: {frame_bgr.shape[:2]}")

        video_writer.write(frame_bgr)

    # Release the video writer
    video_writer.release()
    print(f"Video saved successfully: {output_video_path}")

# Use OpenCV to check the generated video
cap = cv2.VideoCapture(output_video_path)
if not cap.isOpened():
    raise RuntimeError("Generated video cannot be read. Please check the encoder or file integrity!")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Generated video frame count: {frame_count}")
cap.release()