# Leaf Disease Detection using Deep Learning

This project is a leaf disease detection system that uses deep learning techniques, including transfer learning, to identify and classify 33 different types of leaf diseases. The model has been trained on a large dataset of images and is designed to help agricultural professionals and enthusiasts diagnose plant diseases in a fast and accurate manner.

**Dataset**: The model is trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) available on Kaggle. It contains images of various plant diseases for training and evaluation.

![Leaf Disease Detection](leaf-diseases-detect/Media/disease-detecctipn.webp)
![Leaf Disease Detection](leaf-diseases-detect/Media/DanLeaf2.jpg)

## Usage

To use the model for leaf disease detection, follow these steps:
1. Set up a Python environment:
    ```bash
    python -m venv venv
    ```
2. Activate the environment:
    ```bash
    .\venv\Scripts\activate
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the main script:
    ```bash
    streamlit run main.py 
    ```

## Model Details

The leaf disease detection model is built using deep learning techniques and transfer learning. It is trained on a dataset containing images of 33 different types of leaf diseases. For more information about the architecture, dataset, and training process, please refer to the code and documentation provided.
