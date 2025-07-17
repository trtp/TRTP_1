#  A Three-stage Vision-Language Task Planning Framework with Spatial Prompts and Feedback Simulation

**Official Repository of Our Paper: “A Three-stage Vision-Language Robust Task Planning Framework with Spatial Prompts and Feedback Simulation
”**

📄 [Paper (coming soon)]() | 🔗 [Project Page](https://trtp.github.io/)
📌 Note: Paper link will be shared publicly after acceptance.
---

##  Overview

**TRTP** introduces a **three-stage framework** for robust robot task planning. 
Traditional Vision-Language Models (VLMs) often struggle with complex spatial reasoning and lack a mechanism to verify the physical feasibility of their plans. Our framework addresses these limitations by creating a closed-loop system that integrates perception, planning, and simulation-based validation.

The core contributions of our framework are:
1.  **Stage 1: Spatial Prompt Generation:** We use a VLM to analyze a scene's keyframe and generate a rich, descriptive text about object relationships, which we call a **"Spatial Prompt"**.
2.  **Stage 2: Spatially-Aware Task Planning:** This Spatial Prompt is then fed as a system-level context to another VLM, guiding it to produce a more accurate and spatially coherent task plan based on a video instruction.
3.  **Stage 3: Digital Twin Simulation & Feedback:** The generated plan is executed in a high-fidelity digital twin of the environment. If a step fails (e.g., collision, unreachable object), the system generates a structured **"Error Prompt"** that is fed back to the planner, enabling it to iteratively correct and refine its plan until a feasible solution is found.

This repository contains **all components** to reproduce the system, including:
- VLM spatial relation extraction and fusion
- Digital twin environment setup and feedback loop
- Qwen2VL fine-tuning using [`llama-factory`](https://github.com/hiyouga/llama-factory)
- Depth-based mesh generation via [`depth-anything-v2`](https://github.com/isl-org/DPT)
- UE5-based high-fidelity manipulation environment

---

##  Repository Structure


```bash
TRTP_1/
├── VLM_process_Code/        # Stage 1 & 2: Spatial Prompt Extraction, Fusion, and Task Planning
│   └── README.md            # VLM_process_Code's README file
├── DigitalTwinSimEnv/       # Stage 3 (Part 1): 3D Scene Reconstruction from a Single Image
│   └── README.md            # DigitalTwinSimEnv's README file
├── src/                     # Stage 2 (Fine-tuning): Qwen2VL Fine-tuning using llama-factory (LoRA)
├── DataProcess/             # DataProcess code 
│   └── README.md            # DataProcess's README file
├── assets/                  # Images, diagrams, GIFs, and other visual assets
├── requirements.txt         # Core Python dependencies
└── README.md                # This file
```

---

##  Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/trtp/TRTP_1.git
    cd TRTP_1
    ```

2.  **Install core dependencies:**
    A virtual environment is highly recommended.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install specialized dependencies:**
    Our framework leverages several powerful external tools. Please follow their official installation guides:
    -   **For VLM Fine-tuning:** [llama-factory](https://github.com/hiyouga/llama-factory) is required for fine-tuning Qwen2VL.
    -   **For 3D Reconstruction:** [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) models and dependencies are needed. Download checkpoints and place them in `DigitalTwinSimEnv/checkpoints/`.
    -   **For Simulation:** [Unreal Engine 5](https://www.unrealengine.com/en-US) is necessary for running the interactive digital twin environment.


---

##  Modules

### ① VLM Spatial Processing Module (`VLM_process_Code/`)

For more details, see the README file in each subfolder.

This module implements **Stage 1 and 2** of our framework: generating spatial prompts and using them for task planning. It includes a complete pipeline for data processing, inference, and evaluation.

-   **Core Workflow:**
    1.  **Spatial Relation Extraction:** A VLM (e.g., `InternVL2`, `Qwen2VL`) analyzes a keyframe to produce a `.txt` file describing object spatial relations (the "Spatial Prompt").
    2.  **Dataset Construction:** The generated text prompts are paired with their corresponding videos and task instructions to create a structured `.json` dataset.
    3.  **Spatially-Aware Inference:** This JSON dataset is fed to a VLM, which uses the spatial prompt as context to generate a precise task plan.
-   **Evaluation:** Includes scripts to automatically evaluate the quality of the generated spatial prompts (on **Precision, Completeness, Redundancy**) and the final task plans (on **Visual, Temporal, and Physical Consistency**).
-   **Usage Example:**
    ```bash
    cd VLM_process_Code/makeDatasets/llava-onevision-qwen2-7b-ov-hf/

    # Step 1: Generate spatial description texts
    python 1_llava-onevision-qwen2-7b-ov-hfInferImageAndSaveTxt.py

    # Step 2: Create a JSON dataset pairing texts with videos
    python 2_llava-onevision-qwen2-7b-ov-hfMakeDatesetWithSavedTextAndVideos.py

    # Step 3: Perform final task planning using the spatial prompts
    python 3_llava-onevision-qwen2-7b-ov-hfInferTaskByPromptAndVideo.py
    ```

### ② Digital Twin Environment (`DigitalTwinSimEnv/`)

This module implements the **foundation of Stage 3**: automatically reconstructing a 3D environment from a single image. This mesh serves as the geometric basis for our high-fidelity simulation.

-   **3D Reconstruction Pipeline:**
    1.  **Depth Estimation:** `run.py` uses **Depth Anything V2** to generate a high-precision 16-bit depth map from an input image.
    2.  **Point Cloud Generation:** `imageTo3DPoint.py` converts the depth map into a dense 3D point cloud (`.ply`) using camera intrinsics.
    3.  **Mesh Reconstruction:** `Ply2Mesh.py` applies **Poisson Surface Reconstruction** to the point cloud to generate a clean, watertight 3D mesh (`.obj`).
-   **Simulation Integration:** The generated `.obj` mesh is imported into **Unreal Engine 5**, where it is combined with a high-fidelity robot model (e.g., [FAB Manipulator](https://www.fab.com/zh-cn/listings/65192c8a-b0e0-4e8d-8a9c-f2f8b7185f27)) to perform physics-based validation. When a failure occurs, the UE5 environment generates a structured error prompt.

### ③ Model Fine-tuning (`src/`)

This module provides the pipeline for fine-tuning a VLM to better understand task failures and correction instructions.

-   **Framework:** Built on the highly efficient [llama-factory](https://github.com/hiyouga/llama-factory).
-   **Model:** We fine-tune **Qwen2VL** using **LoRA (specifically rsLoRA)** adapters for parameter-efficient training.
-   **Dataset:** The training data consists of pairs of `(video, initial_plan, error_prompt)` and the corresponding `corrected_plan`, teaching the model to re-plan based on simulation feedback.


---



---

## 📌 Citation

```bibtex
@article{TRTP2025,
  title={A Three-stage Vision-Language Robust Task Planning Framework with Spatial Prompts and Feedback Simulation},
  author={},
  journal={},
  year={}
}
```

---

##  License

This repository is licensed under the **MIT License**.

---

##  Acknowledgements
Our work builds upon many fantastic open-source projects. We extend our sincere gratitude to their developers.
- [Qwen2VL](https://github.com/Qwen-VL)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [depth-anything-v2](https://github.com/isl-org/DPT)
- [llama-factory](https://github.com/hiyouga/llama-factory)
- [FAB Robot](https://www.fab.com/zh-cn/listings/65192c8a-b0e0-4e8d-8a9c-f2f8b7185f27)
- [UE5 Digital Twin Engine](https://www.unrealengine.com/)

---

