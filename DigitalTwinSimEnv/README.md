
---

# DigitalTwinSimEnv: Interactive Digital Twin Simulation Environment

This project provides the foundational pipeline for an interactive digital twin simulation environment. The system is designed to automatically construct a 3D virtual scene from a single RGB image. This serves as the first step towards a larger goal of enabling pre-execution feasibility validation and feedback for robotic tasks.

The core functionality currently implemented focuses on the 3D scene reconstruction pipeline, which involves high-fidelity depth estimation and robust mesh generation.

## Key Features (Current)

-   **Single-Image 3D Reconstruction**: Generates a 3D model of a scene from one static RGB image.
-   **High-Fidelity Depth Estimation**: Leverages the **Depth Anything V2** model to produce dense and accurate depth maps.
-   **Point Cloud Generation**: Converts 2D depth maps into a 3D point cloud using camera intrinsic parameters.
-   **Poisson Surface Reconstruction**: Builds a high-quality, watertight 3D mesh from the point cloud, suitable for visualization and future simulation.

## Workflow Pipeline

The reconstruction process is divided into three main stages:

### 1. Depth Estimation
-   **Input**: A single RGB image (e.g., from the `test_image/` directory).
-   **Process**: The `run.py` script utilizes the **Depth Anything V2** model (from the `depth_anything_v2/` module) to predict the scene's geometry.
-   **Output**: The result is a 16-bit depth map (e.g., `.tiff` format), which stores high-precision depth information, and an optional visualization saved in the `vis_depth/` directory.

### 2. Point Cloud Generation
-   **Script**: `imageTo3DPoint.py`
-   **Process**: This script reads the 16-bit depth map. Using provided camera intrinsics (focal length `fx`, `fy` and principal point `cx`, `cy`), it back-projects each pixel's 2D coordinate and depth value into a 3D point in space.
-   **Output**: A dense point cloud, which is saved as a `.ply` file (`output_point_cloud.ply`).

### 3. Mesh Reconstruction
-   **Script**: `Ply2Mesh.py`
-   **Process**:
    1.  Reads the `.ply` point cloud file generated in the previous step.
    2.  Estimates normals for each point, which are crucial for defining surface orientation.
    3.  Applies **Poisson Surface Reconstruction** to generate a continuous, watertight triangular mesh from the discrete points.
    4.  Includes a filtering step to remove vertices in low-density regions, effectively cleaning up artifacts and improving the final mesh quality.
-   **Output**: A high-quality 3D mesh, exported as an `.obj` file (`output_model1.obj`).

## Project Structure

```
DigitalTwinSimEnv/
├── assets/            # (Future) Stores standardized 3D models for object replacement
├── checkpoints/       # Stores pre-trained model weights (e.g., Depth Anything V2)
├── depth_anything_v2/ # Source code/module for the Depth Anything V2 model
├── metric_depth/      # Modules related to metric depth calculation
├── test_image/        # Contains input RGB images for testing
├── vis_depth/         # Output directory for visualized depth maps
├── __init__.py        # Makes DigitalTwinSimEnv a Python package
├── app.py             # (Future) Main application, GUI, or web service entry point
├── imageTo3DPoint.py  # Script to convert a depth map into a 3D point cloud
├── Ply2Mesh.py        # Script to convert a point cloud (.ply) into a 3D mesh (.obj)
├── run.py             # Main script to run depth estimation on a single image
└── run_video.py       # Main script to run depth estimation on a video
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/trtp/TRTP_1.git
    cd TRTP_1
    cd DigitalTwinSimEnv
    ```
    For details on generating depth estimation maps, please refer to https://github.com/DepthAnything/Depth-Anything-V2

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The main dependencies are `opencv-python`, `numpy`, and `open3d`. You will also need `torch`, `timm`, etc., for the `Depth Anything` model.
    ```bash
    pip install opencv-python numpy open3d torch timm
    ```
    *(Note: It is recommended to create a `requirements.txt` file for easier setup.)*

4.  **Download Models:**
    - Download the pre-trained model weights for **Depth Anything V2**.
    - Place the checkpoint files in the `checkpoints/` directory.

## Usage

Follow these steps to generate a 3D model from an image:

1.  **Generate the Depth Map:**
    Run the `run.py` script to process an image from the `test_image/` folder. Ensure the output is a 16-bit `.tiff` file for maximum precision.
    ```bash
    # Example command (adjust arguments as needed)
    python run.py --encoder vitl --img-path test_image/example.jpg --outdir vis_depth
    ```
    *This will generate a `.tiff` depth map inside the `vis_depth` folder.*

2.  **Generate the Point Cloud:**
    The `imageTo3DPoint.py` script converts the generated depth map into a point cloud.
    **Note**: You may need to modify the hardcoded paths (`depth_image_path` and `ply_path`) inside the script.
    ```bash
    python imageTo3DPoint.py
    ```

3.  **Generate the 3D Mesh:**
    The `Ply2Mesh.py` script takes the point cloud and reconstructs a mesh.
    **Note**: This script assumes the input is `output_point_cloud.ply`.
    ```bash
    python Ply2Mesh.py
    ```
    The final model will be saved as `output_model1.obj`.

## Roadmap & Future Enhancements

The current project provides a solid foundation for 3D scene reconstruction. Future work will focus on adding semantic understanding and interactive simulation capabilities to build a complete digital twin system.

-   **Semantic Segmentation**:
    -   Integrate the **Segment Anything Model (SAM)** to perform instance-level segmentation on the input RGB image.
    -   This will allow the system to identify and isolate key objects within the scene (e.g., workbenches, tools, parts).

-   **Rule-Based Object Replacement**:
    -   Develop a system to project the 2D semantic masks onto the 3D mesh.
    -   Create a library of standardized, simulation-ready 3D models in the `assets/` directory.
    -   Implement rules to automatically replace segmented regions of the mesh with their corresponding high-fidelity, interactive 3D models.

-   **Physics-Based Simulation and Feedback**:
    -   Embed a robot model into the final scene.
    -   Use a physics engine to simulate robotic tasks (e.g., pick-and-place).
    -   Implement collision detection and feasibility analysis to validate action sequences.
    -   Develop a feedback mechanism that reports failures back to a task planner, creating a "generate-validate-correct" loop.