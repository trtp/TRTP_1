
---

# Spatial Prompt-Assisted Task Planning

This repository contains the code implementation for the "Spatial Prompt-Assisted Task Planning" section of our research paper. The core idea is to leverage a Vision-Language Model (VLM) to accomplish complex video-based task planning in a two-step process, enhancing the model's spatial awareness and accuracy.

## 1. Project Overview

The project follows a two-step pipeline:

1.  **Spatial Relationship Extraction**: First, a VLM analyzes a key frame from a video to extract the spatial layout and relationships of objects within the scene. This generates a descriptive text, which we term a **"Spatial Prompt"**.
2.  **Task Planning Generation**: Next, this "Spatial Prompt" is fed as context (a System Prompt) along with the original video and a task instruction to the VLM. This guides the model to generate a more accurate and spatially-aware task plan.

This repository provides the complete workflow, including data preprocessing, scripts for generating spatial prompt datasets with various VLMs, and code for evaluating the final task plans.

## 2. Directory Structure

```
VLM_process_Code/
├── dataVideosCollect/                  # 1. Scripts for collecting and processing raw video data
│   ├── Droid_collect_MP4.py
│   ├── RoboMind_H52Videos.py
│   └── RoboMind_H52VideosAll.py
│
├── EvaluationDataset/                  # 2. Scripts for evaluating the final task planning results
│   ├── InternVL2Evaluation.py
│   ├── MiniCPM-V-2_6Evaluation.py
│   └── Qwen2VLEvaluation.py
│
├── EvaluationSpatialRelationshipExtraction/ # 3. Scripts for evaluating spatial relationship extraction capabilities
│   ├── InternVL2Evaluation.py
│   ├── Qwen2VLBaseEvaluation2.py
│   └── temp_frame.jpg
│
├── makeDatasets/                       # 4. Core Module: Generates spatial prompt datasets
│   ├── deepseeklVL2/
│   ├── Emu3/
│   ├── ... (other VLM models)
│   ├── llava-onevision-qwen2-7b-ov-hf/ # <-- Used as an example in the explanation
│   │   ├── 1_llava-onevision-qwen2-7b-ov-hfInferImageAndSaveTxt.py
│   │   ├── 2_llava-onevision-qwen2-7b-ov-hfMakeDatesetWithSavedTextAndVideos.py
│   │   └── 3_llava-onevision-qwen2-7b-ov-hfInferTaskByPromptAndVideo.py
│   │
│   ├── 0_randomGetImage.py             # Utility script to randomly extract frames from videos
│   └── Video_Image_Cut.py              # Utility script for video-to-frame conversion
│
├── outputs/                            # 5. Directory for saving output results (non-code)
│
└── VLMRobotInstruction/                # 6. Scripts for converting task plans into robot instructions
    ├── InternVL2_5RobotController.py
    └── Qwen2VLRobotInstruction.py
```
**Note**: The cross-evaluation validation scripts are currently being refined and will be committed later.

## 3. Core Workflow: Spatial Prompt Dataset Generation

The core of this project lies in the workflow within the `makeDatasets` directory. This process uses a three-step pipeline to construct a video task planning dataset enriched with spatial prompts for a specific VLM (e.g., `llava-onevision-qwen2-7b-ov-hf`).

---

### **Step 1: Generating Spatial Prompt Texts**

-   **Script**: `1_llava-onevision-qwen2-7b-ov-hfInferImageAndSaveTxt.py`
-   **Function**: This script processes key frames from videos to generate descriptive texts about object spatial relationships.
-   **Input**:
    -   A folder containing key frames extracted from various videos (e.g., `droidCutImage_randomGet`).
    -   A fixed text prompt: `message_text = "Describe the spatial relationship of the objects in the scene."`.
-   **Process**: The script iterates through each image, feeding it to the `llava-onevision` model along with the prompt.
-   **Output**:
    -   An output folder (e.g., `droidCutImagePrompt/llava-onevision-qwen2-7b-ov-hf`).
    -   This folder contains `.txt` files, each corresponding to an input image. The content of each file is the VLM-generated description of spatial relationships. **This output is the crucial "Spatial Prompt"**.

---

### **Step 2: Building the JSON Dataset for Inference**

-   **Script**: `2_llava-onevision-qwen2-7b-ov-hfMakeDatesetWithSavedTextAndVideos.py`
-   **Function**: This script associates the generated spatial prompts (`.txt`) with their corresponding original videos (`.mp4`) and structures them into a JSON file.
-   **Input**:
    -   The folder of spatial prompts (`.txt` files) from Step 1.
    -   A folder containing the original videos.
-   **Process**: The script reads each `.txt` file and finds its matching video. It then creates a data structure containing `conversations` and `videos` keys.
    -   The `system` prompt's `value` is set to the content of the `.txt` file (the spatial prompt).
    -   The `human` prompt's `value` is set to the task instruction (e.g., `<video>List the task plan in the video`).
    -   The `gpt` prompt's `value` is temporarily set to "default", awaiting population in the next step.
-   **Output**:
    -   A `.json` file (e.g., `llava-onevision-qwen2-7b-ov-hf-prompt-output_dataset.json`), where each entry contains a **spatial prompt, a task instruction, and a video path**.

---

### **Step 3: Performing Task Planning Inference with Spatial Prompts**

-   **Script**: `3_llava-onevision-qwen2-7b-ov-hfInferTaskByPromptAndVideo.py`
-   **Function**: This script reads the JSON dataset from Step 2, uses the VLM to perform the final task planning inference, and writes the results back into the JSON file.
-   **Input**:
    -   The `.json` dataset file from Step 2.
-   **Process**: The script iterates through each entry in the JSON file.
    -   It combines the **spatial prompt** (`system` prompt), the **video**, and the **task instruction** (`human` prompt) as a composite input for the `llava-onevision` model.
    -   Guided by the spatial context, the VLM analyzes the video and generates a detailed action sequence (task plan).
    -   This generated sequence is used to fill the `value` of the `gpt` field.
-   **Output**:
    -   A final, complete `.json` dataset where the `gpt` field contains the task plan generated with the assistance of the spatial prompt.

**Special Note**: The `3_...py` script is designed flexibly to allow for **cross-model evaluation**. For instance, one can use a dataset with spatial prompts generated by `InternVL2-8B` (input JSON) but perform the final inference with the `llava-onevision` model to test the efficacy of different model combinations.

## 4. How to Run

1.  **Environment Setup**:
    -   Install `PyTorch`.
    -   Install dependencies: `pip install transformers pillow accelerate decord h5py opencv-python`.
    -   Download the required VLM models to a local directory.

2.  **Data Preparation**:
    -   Use `makeDatasets/Video_Image_Cut.py` or `0_randomGetImage.py` to extract key frames from your raw videos.
    -   Update the file and model paths in the scripts to match your local environment.

3.  **Execution Flow**:
    Using `llava-onevision-qwen2-7b-ov-hf` as an example, execute the scripts in order from the `makeDatasets/llava-onevision-qwen2-7b-ov-hf/` directory:
    ```bash
    # Step 1: Generate spatial description texts from images
    python 1_llava-onevision-qwen2-7b-ov-hfInferImageAndSaveTxt.py

    # Step 2: Pair texts with videos to create a JSON dataset
    python 2_llava-onevision-qwen2-7b-ov-hfMakeDatesetWithSavedTextAndVideos.py

    # Step 3: Perform inference using spatial prompts to get the final task plan
    python 3_llava-onevision-qwen2-7b-ov-hfInferTaskByPromptAndVideo.py
    ```

## 5. Detailed Code File Analysis

This section provides a detailed breakdown of the key scripts in the repository.

### 5.1 `dataVideosCollect` - Video Data Collection and Preprocessing

-   **`RoboMind_H52Videos.py` & `RoboMind_H52VideosAll.py`**
    -   **Purpose**: These are tools to convert RoboMind dataset files from `.hdf5` format to standard `.mp4` videos. `...All.py` is a batch-processing version.
    -   **Core Logic**: They use `h5py` to read the `trajectory.hdf5` files, access the image sequence under `observations/rgb_images/camera_top`, and then use `Pillow` and `OpenCV` to decode each frame and write it to a new `.mp4` video file.

### 5.2 `EvaluationDataset` - Task Planning Evaluation

Scripts in this directory use a powerful VLM as a "judge" to automatically evaluate the quality of a generated task plan across three dimensions: **Visual Consistency (S_vision)**, **Temporal Consistency (S_temporal)**, and **Physical Feasibility (S_physical)**.

-   **`InternVL2Evaluation.py`**
    -   **Purpose**: Uses the `InternVL2-8B` model as a judge to batch-evaluate task plans.
    -   **Core Logic**:
        1.  **Multi-frame Input**: Samples 8 frames from the video to provide rich temporal context.
        2.  **Complex Prompting**: Feeds the 8 frames and a detailed prompt defining all three evaluation criteria to the model in a single call.
        3.  **Structured Output Parsing**: Expects the model to return scores in a strict format (`Visual Consistency: <value>...`) and uses regular expressions (`re`) to parse them.
        4.  **Weighted Final Score**: Calculates a final score using predefined weights (`alpha`, `beta`, `gamma`).

-   **`Qwen2VLEvaluation.py` & `MiniCPM-V-2_6Evaluation.py`**
    -   **Purpose**: Use `Qwen2-VL` and `MiniCPM-V-2.6` respectively as judge models.
    -   **Core Logic**:
        1.  **Single-Dimension Evaluation**: These scripts evaluate one criterion at a time. For each sample, they perform three separate inference calls (one for visual, one for temporal, one for physical).
        2.  **Simple Output Parsing**: They expect the model to return a simple float score, which is parsed with `float()`.
        3.  **Weighted Final Score**: The three scores are combined using the same weighted formula.

### 5.3 `EvaluationSpatialRelationshipExtraction` - Spatial Prompt Quality Evaluation

These scripts evaluate the quality of the output from **Step 1** of our pipeline: the spatial prompt text itself. The evaluation metrics are **Precision**, **Completeness**, and **Redundancy**.

-   **`InternVL2Evaluation.py` & `Qwen2VLBaseEvaluation2.py`**
    -   **Purpose**: Use `InternVL2-8B` or `Qwen2-VL` as a judge to assess the quality of the generated spatial prompt text against a video key frame.
    -   **Core Logic**:
        1.  **Input**: A key frame from the video (e.g., the 10th frame) and the system description (the spatial prompt to be evaluated).
        2.  **Evaluation Prompt**: The model is prompted to score the provided text based on its precision, completeness, and redundancy relative to the image. The `Qwen2VL` script uses a particularly detailed prompt that defines each metric for the model.
        3.  **Results Parsing**: The scripts parse the model's output to extract the three scores. They are designed to run on a whole dataset, saving individual results and continuously updating the average scores in a JSON file.