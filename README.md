# Machine Vision and LRF Integration for Railway Obstacle Detection

This repository contains the software architecture and 3D enclosure files for a real-time, zero-latency railway obstacle detection system. It integrates dual physical Laser Range Finders (LRFs) with a multi-threaded AI machine vision pipeline to create a highly redundant, fail-safe hazard warning system for moving trains.

This project was developed as a Capstone Thesis in the Robotics and AI Engineering program at King Mongkut's Institute of Technology Ladkrabang (KMITL).

## 🚀 Key Features
* **Multi-Model AI Pipeline:** Runs YOLOv8 (rail segmentation), YOLO11n (obstacle detection), and MiDaS (monocular depth estimation) simultaneously.
* **Zero-Latency Video Ingestion:** Utilizes `PyAV` to directly decode RTSP network streams, bypassing standard `OpenCV` buffering lag.
* **Asynchronous Audio Alerts:** Uses `pygame` to manage a dedicated audio thread, scaling alarm frequencies dynamically without blocking the main CPU thread.
* **Sensor Fusion Logic:** Continuously evaluates the minimum absolute distance between physical LRFs and the AI-estimated depth map.
* **Optimized Serial Parsing:** Uses a high-speed hexadecimal byte-search algorithm (`0xFB 0x03`) to extract LRF data from USB inputs instantly.

## 💻 Hardware Requirements
* **Processing Unit:** High-performance PC/Laptop with a dedicated CUDA-enabled GPU (Developed on an RTX 3060).
* **Camera:** IP Camera with RTSP streaming capabilities (e.g., Dahua WizSense IPC-HFW2449S-S-IL).
* **Sensors:** * PTFG-3000 Laser Range Finder (3000m range) via Serial-to-USB.
  * PTFS-1000 Laser Range Finder (400m range) via Serial-to-USB.

## 🛠️ Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/job0403/VisionLRFCapstone.git](https://github.com/job0403/VisionLRFCapstone.git)
cd VisionLRFCapstone
```

**2. Install CUDA for PyTorch:**
Because this system runs three deep-learning models simultaneously, **you MUST use the GPU-enabled version of PyTorch**. Standard pip installations may download the CPU version, which will cause the system to run at unplayable framerates.
Please install the correct PyTorch version for your CUDA toolkit from the [official PyTorch website](https://pytorch.org/get-started/locally/).

**3. Install the remaining requirements:**
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration
Before running the code, open `Final_Code.py` and verify the `CONFIGURATIONS` block:
* Update `VIDEO_PATH` to match your camera's RTSP stream IP and credentials.
* Update the COM ports for your specific Serial-to-USB LRF connections inside the LRF thread functions.

## ▶️ Usage
To start the detection system, simply run the main script:
```bash
python Final_Code.py
```
* **To exit the program:** Press `q` in the active display window.
* **Outputs:** If `SAVE_OUTPUT = True` is set in the configuration, the system will automatically create an `output/` directory and save the annotated video there.

## 🙏 Special Thanks & Acknowledgements
* **[Ultralytics (YOLO)](https://github.com/ultralytics/ultralytics):** A massive thank you to the Ultralytics team for the YOLOv8 and YOLO11n architectures, which powered the real-time track segmentation and hazard detection in this project.
* **[MiDaS](https://github.com/isl-org/MiDaS):** Special thanks to the ISL team for their incredibly robust monocular depth estimation models and source code, which allowed our system to maintain spatial awareness and depth perception using a single camera.

## 👨‍💻 Authors
* **Pongpiched Rakshit**
* **Supichaya Kritpidhayaburana**
* **Advisors:** Assoc. Prof. Somyot Kaitwanidvilai, Ms. Mathinee Songthai (KMITL)