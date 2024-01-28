## Advanced Vehicle Detection and Counting System

## Overview

This project is aimed at developing an advanced vehicle detection and counting system for traffic analysis and management. Utilizing sophisticated image processing and machine learning techniques, this system can detect vehicles in video streams and accurately count their number as they pass through predefined checkpoints. The project is implemented using Python, with key libraries such as OpenCV for image processing, Scikit-learn for machine learning models, and Pandas for data handling.

## Key Features

- **Vehicle Detection:** Ability to detect vehicles in video frames using machine learning and image processing techniques.
- **Cross Line Detection:** Detects when a vehicle crosses a predefined line (e.g., a stop line at a traffic signal) and counts the vehicle.
- **Accuracy Assessment:** Implements Mean Absolute Error (MAE) calculation for evaluating the accuracy of the vehicle counting against true values.
- **Image and Video Processing:** Uses OpenCV for processing and analyzing video streams.
- **Machine Learning Model:** Utilizes Support Vector Machine (SVM) for vehicle classification.

## Dependencies

- Python 3.x
- OpenCV
- Scikit-learn
- Pandas
- NumPy

## Installation

Ensure Python is installed on your system. Install the required libraries using the following command:

```
pip install opencv-python-headless scikit-learn pandas numpy
```

## Usage

To use this system, place your video files in the 'videos' directory and run the Python script:

```
python [ScriptName].py [FolderPath]
```

Replace `[ScriptName].py` with the name of the Python script and `[FolderPath]` with the path to the folder containing your videos.

## File Structure

- `[ScriptName].py`: Main Python script for vehicle detection and counting.
- `videos/`: Directory containing video files for processing.
- `pictures/`: Directory containing images used for training the SVM model.

## How It Works

1. **Vehicle Detection:** The system processes each frame of the video, using HOG features and an SVM classifier to detect vehicles.
2. **Counting Algorithm:** When a vehicle crosses the predefined line, it is counted. The system keeps track of the count throughout the video.
3. **Accuracy Calculation:** The system calculates the MAE between the predicted vehicle count and the actual count (provided in a CSV file).

## Contributing

We welcome contributions to improve the system. Feel free to fork the repository, make your changes, and submit a pull request.


