
---

#  Multimodal Model Evaluation Results Plot &  Datasets Exact Toolkit


---

##  Project Structure

```bash
.
├── CrossValidation.py                  # Visualizes model cross-evaluation scores as a heatmap
├── PlotJson2Loss2.py      # Compares loss curves for models with and without prompts
├── clvr_jaco_play_dataset_H52Videos.py                # Extracts image sequences from H5 files and generates videos (with prompts)
├── BridgeData_PickAndTransferImg2Video.py              # Combines image sequences from a folder into a video
├── data/                           # Folder for JSON loss data
├── output/                         # Directory for saving visualization outputs
```

---

##  Score Heatmap: `CrossValidation.py`

Generates a heatmap of cross-evaluation scores between models, with an option to mask the diagonal (self-scores).

###  Features:

- Reads a predefined matrix (score data).
- Uses seaborn to plot the heatmap.
- Masks diagonal scores with a gray overlay to avoid self-evaluation bias.
- Supports output in `.pdf`, `.svg`, and `.png` formats.

###  Output:

- `CrossValHeatMap.pdf`
- `CrossValHeatMap.svg`
- `CrossValHeatMap.png`

---

##  Loss Curve Comparison Plot: `plot_promptloss_compare.py`

Visualizes and compares the loss curves of models with and without prompts. Automatically adds annotations and prevents them from overlapping.

###  Features:

- Supports batch loading of multiple loss datasets (in JSON format).
- Automatic annotation box layout to prevent overlapping.
- Compares a trained model against a baseline/control model.
- Supports vector graphics output in PDF and SVG formats.

###  Input Data Structure:

```bash
data/
├── *_promptloss_data.json          # Loss data with prompts
├── *_inferloss_data.json           # Control group (no prompt) loss data
```

###  Output:

- `output/loss_comparison_plot.svg`
- `output/loss_comparison_plot.pdf`

---

##  HDF5 Image Sequence to Video: `video_from_h5.py`

Reads image sequences (`front_cam_ob`) from an `.h5` dataset, segments them into behavioral episodes based on `terminals`, exports videos per skill, and records the corresponding prompts.

###  Features:

- Saves each behavioral episode as a separate `.mp4` video.
- Writes the prompt for each episode to `prompts.txt`.
- Automatically creates the output directory if it doesn't exist.

###  Output:

- Video files: `videos_all_play_data_diverse/skill_*.mp4`
- Text file: `videos_all_play_data_diverse/prompts.txt`

---

##  Batch Convert Image Sequences to Video: `images_to_video.py`

Scans for `images6` folders within a specified directory, automatically combining the image sequences into videos. Ideal for processing scripted datasets.

###  Features:

- Recursively searches for `images6` folders.
- Reads images and compiles them into an `.mp4` video.
- Supports natural sorting of image files.



##  Dependencies

Please ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib seaborn opencv-python h5py
```

---
