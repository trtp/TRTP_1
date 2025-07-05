# import os
# import shutil
#
# def extract_and_rename_videos(source_dir, target_dir, extensions):
#     """
#     Batch extract and rename video files (by moving them directly).
#
#     :param source_dir: The source directory to search for video files.
#     :param target_dir: The target directory where video files will be moved.
#     :param extensions: A list of video file extensions (e.g., [".mp4", ".avi"]).
#     """
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     file_counter = 1  # Used to generate unique filenames
#
#     # Traverse the source directory and all its subdirectories
#     for root, _, files in os.walk(source_dir):
#         for file in files:
#             # Check if the file extension is one of the specified video formats
#             if any(file.lower().endswith(ext) for ext in extensions):
#                 source_path = os.path.join(root, file)
#                 # Rename the file using a unique sequence number
#                 new_name = f"video_{file_counter}{os.path.splitext(file)[1]}"
#                 target_path = os.path.join(target_dir, new_name)
#
#                 # Move and rename the file
#                 shutil.move(source_path, target_path)
#                 print(f"Extracted and moved: {source_path} -> {target_path}")
#
#                 file_counter += 1
#
# # Example usage
# source_directory = "/home/ubuntu/Desktop/dataset/collection-droid"  # Replace with your source directory path
# target_directory = "/home/ubuntu/Desktop/dataset/droidMP4"  # Replace with your target directory path
# video_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # Add video formats to support
#
# extract_and_rename_videos(source_directory, target_directory, video_extensions)


import os
import shutil


def extract_videos_in_batches(source_dir, target_dir, extensions, batch_size):
    """
    Extracts videos from a folder into subfolders, with each subfolder
    containing up to 'batch_size' videos (by moving them directly).

    :param source_dir: The source directory to search for video files.
    :param target_dir: The target directory where video files will be extracted to.
    :param extensions: A list of video file extensions (e.g., [".mp4", ".avi"]).
    :param batch_size: The maximum number of video files per target subfolder.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_counter = 0  # Global counter for the total number of video files
    batch_counter = 1  # Batch counter for creating new subfolders
    current_batch_dir = os.path.join(target_dir, f"batch_{batch_counter}")
    os.makedirs(current_batch_dir, exist_ok=True)

    # Traverse the source directory and all its subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file extension is one of the specified video formats
            if any(file.lower().endswith(ext) for ext in extensions):
                source_path = os.path.join(root, file)

                # If the current batch reaches the size limit, create a new subfolder
                if file_counter > 0 and file_counter % batch_size == 0:
                    batch_counter += 1
                    current_batch_dir = os.path.join(target_dir, f"batch_{batch_counter}")
                    os.makedirs(current_batch_dir, exist_ok=True)

                # Generate the file path in the current batch folder
                new_name = f"video_{file_counter + 1}{os.path.splitext(file)[1]}"
                target_path = os.path.join(current_batch_dir, new_name)

                # Move and rename the file
                shutil.move(source_path, target_path)
                print(f"Extracted and moved: {source_path} -> {target_path}")

                file_counter += 1


# Example usage
source_directory = "/home/ubuntu/Desktop/dataset/droidMP4"  # Replace with your source directory path
target_directory = "/home/ubuntu/Desktop/dataset/droid"  # Replace with your target directory path
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # Add video formats to support
batch_size = 1000  # The maximum number of videos in each subfolder

extract_videos_in_batches(source_directory, target_directory, video_extensions, batch_size)