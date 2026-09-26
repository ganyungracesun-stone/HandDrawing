# Hand Tracking Projects

This repository contains several independent Python projects using **MediaPipe Hand Landmarker**.

Each project demonstrates a different application of hand tracking. The projects share the same hand landmark model, but each Python program can be run independently.

## Projects

### 1. Hand Drawing

Use one hand and fingertip movements to draw on the screen.

**Run:**

```bash
python draw_main.py
```

### 2. Gesture Music

Use different hand gestures to play different piano notes.

**Run:**

```bash
python gesture_music.py
```

### 3. Gesture Sign

Use hand gestures to control random letters into a sign (heart, sad-face, fish-eating, stop-sign).

**Run:**

```bash
python gesture_sign.py
```

### 4. Test Camera Demo

A basic demonstration of testing computer camera.

**Run:**

```bash
python test_cam.py
```

## Requirements

* Python 3.11 or later
* PyCharm or another Python IDE
* A computer with a webcam
* Git
* The Python packages listed in `requirements.txt`

## Installation

### 1. Clone the repository

In PyCharm, select:

**Git → Clone**

Enter the repository URL and select a location on your computer.

### 2. Create a virtual environment

PyCharm can create a `.venv` automatically when setting up the project.

### 3. Install the required packages

Open the PyCharm Terminal and run:

```bash
pip install -r requirements.txt
```

## Model

The projects use the MediaPipe Hand Landmarker model:

```text
hand_landmarker.task
```

The model is shared by the different projects in this repository.

## Running a Project

Each project is independent.

Open the Python file you want to run in PyCharm and click the **Run** button.

For example:

```bash
python draw_main.py
```

You do not need to run the other projects.

## Project Structure

```text
HandTrackingProjects/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── model/
│   └── hand_landmarker.task
│
├── hand_drawing/
│   └── draw_main.py
│
├── gesture_music/
│   └── gesture_music.py
│
├── gesture_sign/
│   └── gesture_sign.py
│
└── test_cam/
    └── test_cam.py
```

## Notes

* Make sure your webcam is connected and available.
* Allow the Python program to access the webcam if Windows asks for permission.
* Only one program should normally use the webcam at a time.
* Keep `hand_landmarker.task` in the `models` folder unless you update the model path in the Python code.

## License

This project is licensed under the MIT License.
You are free to use, copy, modify, and distribute the code.
