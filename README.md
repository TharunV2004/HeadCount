# People HeadCount using YOLOv8

This project detects and tracks people in a video and counts movement across a horizontal line.

## What you need (Prerequisites)

Install these first:

- Python 3.10 or newer
- `pip` (comes with Python)
- Git (optional, only if cloning from GitHub)
- A webcam or an MP4 input video

## Project installation

### 1) Clone the repository

```bash
git clone https://github.com/TharunV2004/HeadCount.git
cd HeadCount
```

### 2) Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Required project files

Before running, make sure these files are available:

- `yolov8x.pt` in the project root (same folder as `main.py`)
- Input video in `Input/input.mp4`
  - If `Input/input.mp4` is missing, the script uses the first `.mp4` found in `Input/`

## Run the project

```bash
python main.py
```

- Press `Esc` to stop.
- Output video will be saved as `Final_output.mp4`.

## Notes

- If Python command is not found, try `python3 main.py`.
- If OpenCV window does not open on some Linux setups, install system GUI dependencies for OpenCV.
