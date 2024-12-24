# Image Classifier WebApp

## Overview
This project is a web application for classifying images, built using **Streamlit**. The app enables users to upload an image and get predictions based on a pre-trained machine learning model. (ResNet50 CNN)

## Features
- **Image Upload:** Drag and drop or browse to upload images.
- **Real-time Predictions:** Classify uploaded images instantly.
- **User-friendly Interface:** Built with Streamlit for an intuitive experience.

## Usage
1. Open the app in your browser (default: https://webapp-img-classifier.streamlit.app/).
2. Upload an image using the provided upload button.
3. View the classification results displayed on the screen.

## File Structure
- `ResNet50-streamlit.py`: Main Streamlit application file.
- `ResNet50-api.py`: API containing the pre-trained model.
- `README.md`: Documentation.

## Dependencies
API:
- FastAPI
- Uvicorn
- TensorFlow
- Keras
- PIL (Python Imaging Library)
- NumPy

Streamlit app:
- Streamlit
- requests


## Model Details
The app uses a pre-trained model fine-tuned on 1000 image categories from the [https://en.wikipedia.org/wiki/ImageNet](ImageNet dataset) including:
- tench
- goldfish
- great white shark
+ 997 others


## License
This project is licensed under the [MIT License](LICENSE).