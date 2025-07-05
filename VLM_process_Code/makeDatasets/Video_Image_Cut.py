import os
import cv2


def extract_frame_from_videos(source_dir, target_dir, extensions, frame_time=1):
    """
    Extracts one frame from each video file and saves it to a target directory.

    :param source_dir: The source directory containing video files.
    :param target_dir: The target directory to save the extracted frames.
    :param extensions: A list of video file extensions (e.g., [".mp4", ".avi"]).
    :param frame_time: The time to capture the frame (in seconds, defaults to the 1st second).
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    video_counter = 0  # Video counter

    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file extension is one of the specified video formats
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]

                # Open the video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video file: {video_path}")
                    continue

                # Set the video reading position (by time)
                fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
                if fps == 0:
                    print(f"Could not get FPS for video: {video_path}. Skipping.")
                    cap.release()
                    continue

                frame_number = int(fps * frame_time)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                # Read the frame
                success, frame = cap.read()
                if success:
                    # Save the frame as an image
                    output_path = os.path.join(target_dir, f"{video_name}.jpg")
                    cv2.imwrite(output_path, frame)
                    print(f"Extracted frame to: {output_path}")
                else:
                    print(f"Could not extract a frame from {video_path} at {frame_time}s")

                # Release the video object
                cap.release()
                video_counter += 1

    print(f"Processed a total of {video_counter} video files.")


# Example usage
source_directory = "/home/ubuntu/Desktop/dataset/droid"  # Replace with the source directory path containing video files
target_directory = "/home/ubuntu/Desktop/dataset/droidCutImage"  # Replace with the target directory path to save extracted frames
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # Supported video formats
frame_time = 1  # Capture from the 1st second

extract_frame_from_videos(source_directory, target_directory, video_extensions, frame_time)