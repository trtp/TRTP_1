# -*- coding: utf-8 -*-
import h5py
import numpy as np
import cv2
import os

# Configure paths and parameters
h5_file_path = "E:/code/DataSets/all_play_data_diverse/all_play_data_diverse.h5"
output_video_dir = "videos_all_play_data_diverse"
prompt_file_path = os.path.join(output_video_dir, "prompts.txt")
os.makedirs(output_video_dir, exist_ok=True)  # Create folder to store videos

# Open h5 file
with h5py.File(h5_file_path, "r") as F:
    # Extract data
    data = {key: np.array(F[key]) for key in F.keys()}

    # Open text file for writing
    with open(prompt_file_path, "w") as prompt_file:
        # Iterate through terminals to split skill sequences
        start_idx = 0
        skill_count = 0
        for i, is_terminal in enumerate(data['terminals']):
            if is_terminal:
                # Extract image sequence for current skill
                sequence_images = data['front_cam_ob'][start_idx:i + 1]
                prompt = data['prompts'][skill_count]  # Extract corresponding prompt

                # Set output video path
                video_name = f"skill_{skill_count}.mp4"
                video_path = os.path.join(output_video_dir, video_name)

                # Save as video
                height, width, _ = sequence_images[0].shape
                fps = 10  # Video frames per second
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                for frame in sequence_images:
                    bgr_frame = frame[..., ::-1]  # Convert RGB to BGR (cv2 format)
                    video_writer.write(bgr_frame)

                video_writer.release()
                print(f"Saved video: {video_path}")

                # Write prompt-video correspondence
                prompt_file.write(f"{video_name}: {prompt}\n")
                print(f"Saved prompt: {prompt}")

                # Update index and count
                start_idx = i + 1
                skill_count += 1

print(f"All prompts saved to {prompt_file_path}")