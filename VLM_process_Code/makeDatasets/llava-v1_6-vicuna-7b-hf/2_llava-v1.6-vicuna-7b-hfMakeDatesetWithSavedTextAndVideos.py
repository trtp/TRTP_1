import os
import json


def get_all_videos(video_folder):
    """
    Recursively gets the paths of all video files in a video folder.
    :param video_folder: The path to the video folder.
    :return: A dictionary containing video filenames and their full paths.
    """
    video_files = {}
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                video_files[file_name] = file_path
    return video_files


def create_dataset(txt_folder, video_folder, output_file):
    """
    Generates a dataset from a folder of TXT files and a folder of videos.
    :param txt_folder: The path to the folder containing TXT files with scene information.
    :param video_folder: The path to the folder containing video files.
    :param output_file: The save path for the output dataset.
    """
    dataset = []
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    video_files = get_all_videos(video_folder)

    for txt_file in txt_files:
        # Get the filename (without extension)
        name = os.path.splitext(txt_file)[0]

        # Remove the "_frame" suffix from the filename if it exists
        name = name.replace("_frame", "")

        # Find the corresponding video
        video_path = video_files.get(name)
        if not video_path:
            print(f"Corresponding video not found for: {name}")
            continue

        # Read the content of the TXT file
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            scene_info = f.read().strip()

        # Construct the data item
        data_item = {
            "conversations": [
                {
                    "from": "system",
                    "value": scene_info  # Use the scene description from the TXT file as the 'system' value
                },
                {
                    "from": "human",
                    "value": "<video>List the task plan in the video"
                },
                {
                    "from": "gpt",
                    "value": "default"
                }
            ],
            "videos": [video_path]
        }
        dataset.append(data_item)
        print(f"Processed: {txt_file} and {video_path}")

    # Save the dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Dataset has been saved to: {output_file}")


# Example usage
txt_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/llava-v1.6-vicuna-7b-hf"  # Replace with your TXT folder path
video_folder = "/home/ubuntu/Desktop/dataset/droid"  # Replace with your video folder path
output_file = "/home/ubuntu/Desktop/dataset/droidJsonDatset/llava-v1.6-vicuna-7b-hf-prompt-output_dataset.json"  # Replace with your output dataset path

create_dataset(txt_folder, video_folder, output_file)