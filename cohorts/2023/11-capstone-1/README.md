# MRI Image Labelling Project

## Overview

This project focuses on the automated labelling and plane differentiation of MRI images using a deep learning model. The objective is to streamline the process of generating image labels for patient MRI scans. The project includes components for downloading images from Azure Blob Storage, processing them through a pre-trained MRI model, and storing the resulting labels.

## Table of Contents

- [Files](#files)
- [Usage](#usage)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Files

### [main.py](main.py)

This file contains the main script for downloading patient images from Azure Blob Storage, processing them using an MRI model, and generating image labels. The script utilizes the `prefect` library for workflow automation. It includes error handling and logs the progress of the image processing.

### [labeller.py](labeller.py)

A module for labelling MRI images. It defines a Prefect flow named "Get Images" that downloads patient images from Azure Blob Storage, processes them using a pre-trained MRI model, and generates labels. The generated labels are stored in a JSON file (`labels.json`). The module also uses the `prefect_azure` library for Azure Blob Storage interactions.

### [model.py](model.py)

This file contains the architecture of the MRI model. The model is a convolutional neural network (CNN) designed for image processing. It utilizes the PyTorch framework and includes convolutional layers, max-pooling, dropout layers, and fully connected layers.

### [env.example](env.example)

An example of the details of environment file containing configuration variables, including Azure Blob Storage credentials and container names.

### [requirements.txt](requirements.txt)

A list of required Python packages and dependencies for the project, including FastAPI, Pandas, MLflow, Hyperopt, python-decouple, Prefect, and Uvicorn.

### [Dockerfile](Dockerfile)

A Dockerfile for containerizing the application. It specifies the base image, installs the required packages, and sets up the necessary environment. The file also includes commands for exposing the port and starting the application using Gunicorn.

### [startapp.sh](startapp.sh)

A bash script to start the Gunicorn server for running the FastAPI application within the Docker container.

### [brain-mri-classification-object-detection.ipynb](brain-mri-classification-object-detection.ipynb)

A Jupyter notebook documenting the process of building the MRI model, including model architecture definition, training, and saving model weights.

## Usage

To run the project, follow these steps:

1. Set up the necessary environment variables in the `.env`.
2. Build the Docker image using the provided Dockerfile:
    ```bash
    docker build -t mri-labeller .
    ```
3. Run the Docker container:
    ```bash
    docker run -p 8000:8000 --env-file .env mri-labeller
    ```

The FastAPI application will be accessible at [http://localhost:8000](http://localhost:8000).

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Joseun/machine-learning-zoomcamp/cohorts/2023/11-capstone-1.git
cd 11-capstone-1
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dependencies

Key dependencies for this project include:

- FastAPI
- Pandas
- MLflow
- Hyperopt
- python-decouple
- Prefect
- Uvicorn

## Configuration

Ensure that the necessary environment variables are configured in the `.env`. These variables include Azure Blob Storage credentials (`BLOB_CONNECTION_STRING`) and the container name (`CONTAINER_NAME`).

## Contribution

Contributions to the project are welcome. If you encounter issues or have suggestions for improvements, please open an issue or submit a pull request.


## Acknowledgments

This project utilizes MRI image datasets for brain tumor object detection. The datasets were downloaded from the Kaggle repository available at [Brain Tumor Object Detection Datasets](https://www.kaggle.com/davidbroberts/brain-tumor-object-detection-datasets/). We acknowledge and appreciate the contribution of the dataset provider for making this valuable resource available. 
This project referenced articles and sources. See the "References" section in the README for further details.

## References

This project may reference the following articles and sources:
- [Brain Tumor MRI Single Object Detection](https://www.kaggle.com/code/alirezasoltanikh/brain-tumor-mri-single-object-detection): Inspiration for this MRI Image Labelling Project.
- [Image Classification in Pytorch](https://github.com/vedaant-varshney/ImageClassifierCNN/blob/master/Image%20Classifier.ipynb): Building classification models with Pytorch.
- [Training a Classifier in Pytorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html): Useful resources on PyTorch.

Feel free to reach out to the project maintainers if you have any questions or need further assistance.

