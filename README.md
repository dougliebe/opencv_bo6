# opencv_bo6
starter code for killfeed reading in BO6

# OpenCV OCR Project

## Overview
This project uses OpenCV to read video frames, preprocess them, and run OCR to extract information from specific areas. The extracted information is then formatted and saved.

## Instructions
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the main script using `python main.py`.

## Project Structure
- `requirements.txt`: List of dependencies.
- `main.py`: Main script to run the project.
- `src/`: Source files.
  - `video_reader.py`: Code to read and handle video frames.
  - `ocr_processor.py`: Code to preprocess frames and run OCR.
  - `frame_analyzer.py`: Code to identify and extract information from specific areas.
  - `output_writer.py`: Code to format and save output.
- `output/`: Folder for storing output files.