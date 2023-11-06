# Amazon Product Recommendation System

This project focuses on building a recommendation system for Amazon products using a dataset of Amazon product reviews. The goal is to provide personalized product recommendations to users based on their preferences and historical interactions with Amazon products. The project also includes a web server built with FastAPI, allowing the infrastrucute to generate personalized recommendations from a userID.

## Table of Contents
- [Data](#data)
- [Notebook](#notebook)
- [Training](#training)
- [Prediction](#prediction)
- [Reproducibility](#reproducibility)
- [Deployment](#deployment)
- [Dependencies](#dependencies)
- [Containerization](#containerization)

## Data
The dataset used for this project is available in two files:
- `amazonproducts.parquet`: This file contains information about Amazon products, including product details and reviews.
- `recommendations.parquet`: This file contains precomputed product recommendations for known users.

To obtain the dataset, you can follow the instructions in the [data/README.md](data/README.md) file or download it from [source_link](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) or [source_link](https://jmcauley.ucsd.edu/data/amazon/).

## Notebook
The Jupyter notebook named `ranking.ipynb` provides a detailed exploration of the data preparation, cleaning, EDA, feature importance analysis, model selection, and parameter tuning. It offers a step-by-step explanation of the entire process.

## Training
The `train.py` script is responsible for training the final recommendation model. It loads the data, performs model training, and saves the trained model to a file, such as a Pickle file or a specialized format like BentoML.

## Prediction
The `main.py` script loads the generated recommendations and serves it as a web service using FastAPI. The frontend/backend infrastructure can provide their USERID, and the script will return personalized product recommendations based on the trained model.

## Reproducibility
To ensure reproducibility, a `requirements.txt` file is included in the project, listing all the Python packages and dependencies required for the project. You can create a virtual environment and install these dependencies using `pip` or use a package manager like Pipenv if preferred.

## Containerization
The project includes a Dockerfile (`Dockerfile`) to create a Docker image for the recommendation system. The Docker image ensures that the environment and dependencies are consistent when deploying the service.

## Deployment
The deployment of the recommendation system can be done using Docker containers on any suitable platform or cloud service.

Feel free to reach out to the project maintainers if you have any questions or need further assistance.
