# Monocular 3D Object Detection & Metric Depth Estimation

**Course:** 3D Perception & Learning-Based Data Fusion (A.Y. 2026/2027) - University of Parma  

---

## 1. Overview
This repository contains the Python/PyTorch implementation for **Project 1 (3D Perception)**. 
The pipeline performs real-time 2D semantic detection (**YOLOv8m**) and dense monocular depth estimation (**MiDaS**), extracting metric distances via central ROI median sampling and stabilizing measurements with an Exponential Moving Average (EMA) low-pass filter.

---

## 2. Project Structure
* `test.py` — Main perception pipeline (supports both `.mp4` videos and raw image sequences).
* `requirements.txt` — List of required Python dependencies.
* `.gitignore` — Excludes virtual environments, heavy datasets, pre-trained weights, and video files.

---

## 3. Dataset, Videos & Presentation Material
Due to GitHub file size limits, the input test videos, the rendered split-screen output video (`output_dimostrazione.mp4`), the technical report, and the presentation slides are hosted on the cloud at the following link:

**[Download Videos, Report & Slides (Google Drive / OneDrive)](https://univpr-my.sharepoint.com/:f:/g/personal/arashpreet_singh_studenti_unipr_it/IgAE0jMhY3gxR4TUWjIqPIvuAQess9RQ5cbHtm0_FJkuIbs?e=fkXCiT)**

### How to set up the data:
* **Video Mode:** Download `test_video.mp4` from the cloud link above, create a folder named `data/` inside the project root directory, and place the file inside it (`data/test_video.mp4`).
* **Image Sequence Mode (KITTI):** Download the *Left Color Images* from the official KITTI Vision Benchmark Suite, extract the frames into `data/kitti/training/image_2/`, and set `modalita = "foto"` inside `main()` in `test.py`.

---

## 4. Installation & Execution

1. **Create and activate a virtual environment (Windows PowerShell):**

    python -m venv venv
    .\venv\Scripts\Activate.ps1

2. **Install the required libraries:**

    pip install -r requirements.txt

3. **Run the perception pipeline:**

    python test.py

*(Note: Pre-trained weights for YOLOv8m and MiDaS will be downloaded automatically on the first run).*
