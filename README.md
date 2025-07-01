#  A Three-stage Vision-Language Task Planning Framework with Spatial Prompts and Feedback Simulation

**Official Repository of Our Paper: “A Three-stage Vision-Language Task Planning Framework with Spatial Prompts and Feedback Simulation
”**

📄 [Paper (Manuscript)](./VLP基本框架4.docx) | 🔗 [Project Page](https://github.com/trtp/TRTP_1)

---

## 🧩 Overview

**TRTP** introduces a **two-stage framework** for robust robot task planning. It integrates:
1. ✴️ **Spatial prompt modeling** using VLMs (e.g., Qwen-VL, InternVL);
2. 🧠 **Digital twin simulation with error feedback**, enabling iterative correction;
3. ⚙️ **Closed-loop planning pipeline** grounded in visual, spatial, and physical consistency.

This repository contains **all components** to reproduce the system, including:
- VLM spatial relation extraction and fusion
- Digital twin environment setup and feedback loop
- Qwen2VL fine-tuning using [`llama-factory`](https://github.com/hiyouga/llama-factory)
- Depth-based mesh generation via [`depth-anything-v2`](https://github.com/isl-org/DPT)
- UE5-based high-fidelity manipulation environment

---

## 📂 Repository Structure

```bash
TRTP_1/
├── VLM_process_Code/        # Section 3.1 & 3.2: Prompt Extraction & Fusion
├── DigitalTwinSimEnv/       # Section 3.3: Digital Twin + Depth Mesh Generation
├── src/                     # Qwen2VL fine-tuning using llama-factory (LoRA)
├── assets/                  # Images, diagrams, visualizations
├── requirements.txt         # Core dependencies
└── README.md
```

---

## 🔧 Setup & Installation

```bash
# Clone the repo
git clone https://github.com/trtp/TRTP_1.git
cd TRTP_1

# Install core dependencies
pip install -r requirements.txt
```

You may also need:
- [llama-factory](https://github.com/hiyouga/llama-factory) (for fine-tuning Qwen2VL)
- [depth-anything-v2](https://github.com/isl-org/DPT) (for depth-to-mesh conversion)
- [Unreal Engine 5](https://www.unrealengine.com/) (for high-fidelity simulation)

---

## 🔨 Modules

### ① VLM Process Module (`VLM_process_Code/`)

Implements:
- Spatial relation extraction (Sec 3.1)
- Cross-model scoring and prompt fusion (Sec 3.2)

Used VLMs: **InternVL**, **Qwen2VL**, **OpenFlamingo**

Run:
```bash
cd VLM_process_Code
python extract_relations.py --image examples/scene1.png
python fusion_module.py --inputs scene1_relations.json
```

### ② Digital Twin Environment (`DigitalTwinSimEnv/`)

Implements:
- Depth-based mesh generation using `depth-anything-v2`
- Grid object layout for simulation
- Structured error prompt generation

> **Note**: Full mesh-based simulation is implemented in **Unreal Engine 5**, where structured prompts are imported via JSON interface.

🔧 Robot model: [Manipulator Robot from FAB](https://www.fab.com/zh-cn/listings/65192c8a-b0e0-4e8d-8a9c-f2f8b7185f27)

### ③ Model Fine-tuning (`src/`)

- Based on [llama-factory](https://github.com/hiyouga/llama-factory)
- Fine-tuning pipeline uses **Qwen2VL** with **LoRA (rsLoRA)** adapters
- Sample data: vision-language pairs with spatial prompts and failure recovery traces

Train (simplified):

```bash
cd src
bash finetune_qwen2vl.sh
```

---

## 🧪 Evaluation

We evaluate TRTP on:
- 5 real-world scene tasks
- 3 baseline LMPs: SayCan, ProgPrompt, InnerMonologue
- Metrics: task success rate, spatial consistency, physical feasibility

Full results in [`VLP基本框架4.docx`](./VLP基本框架4.docx)

---

## 🎥 Simulation & Visualization

<p align="center"><img src="assets/trtp_sim_loop.gif" width="600"></p>

Structured Error Prompt (example):

```json
{
  "Failed Step": "place cup",
  "Reason": "Unreachable due to occlusion",
  "Suggested Fix": "Reorder task or reposition object"
}
```

---

## 📌 Citation

```bibtex
@article{TRTP2025,
  title={Two-Stage Robust Task Planning via Structured Spatial Prompts and Digital-Twin Feedback},
  author={Your Name and Co-authors},
  journal={TBD},
  year={2025}
}
```

---

## 🧾 License

This repository is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- [Qwen2VL](https://github.com/Qwen-VL) by Alibaba DAMO
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [depth-anything-v2](https://github.com/isl-org/DPT)
- [llama-factory](https://github.com/hiyouga/llama-factory)
- [FAB Robot](https://www.fab.com/zh-cn/listings/65192c8a-b0e0-4e8d-8a9c-f2f8b7185f27)
- [UE5 Digital Twin Engine](https://www.unrealengine.com/)

---

📫 For questions or contributions: [your_email@domain.com]
