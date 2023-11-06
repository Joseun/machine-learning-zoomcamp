# Amazon Product Recommendation System

This project focuses on building a recommendation system for Amazon products using a dataset of Amazon product reviews. The goal is to provide personalized product recommendations to users based on their preferences and historical interactions with Amazon products. The project also includes a web server built with FastAPI, allowing the infrastructure to generate personalized recommendations from a user identification parameter.

## Table of Contents
- [Data](#data)
- [Notebook](#notebook)
- [Training](#training)
- [Prediction](#prediction)
- [Reproducibility](#reproducibility)
- [Deployment](#deployment)
- [Improvements](#improvements)
- [Contributions](#contributions)
- [Tools Used](#tools-used)
- [References](#references)

## Data
The dataset used for this project is available in two files:
- `reviews.csv`: This file contains information about Amazon products, including product details and reviews.
- `recommendations.parquet`: This file contains precomputed product recommendations for known users.

To obtain the dataset, you can download it from [Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) or [archive source](https://jmcauley.ucsd.edu/data/amazon/).

## Notebook
The Jupyter notebook named `ranking.ipynb` provides a detailed exploration of the data preparation, cleaning, EDA, feature importance analysis, model selection, and parameter tuning. It offers a step-by-step explanation of the entire process.

## Training
The `preprocess.py`, and `train.py` script is responsible for preprocessing and training the final recommendation model. It loads the data, performs model training, and saves the trained model to a file, such as a Pickle file.

## Prediction
The `main.py` script loads the generated recommendations and serves it as a web service using FastAPI. The frontend/backend infrastructure can provide their USERID, and the script will return personalized product recommendations based on the trained model.

## Reproducibility
To ensure reproducibility, a `requirements.txt` file is included in the project, listing all the Python packages and dependencies required for the project. You can create a virtual environment and install these dependencies using `pip` or use a package manager like Pipenv if preferred.

## Deployment
The deployment of the recommendation system can be done using Docker containers on any suitable platform or cloud service.

## Improvements
### Lessons Learned
During the project, several valuable lessons were learned, including:
- The importance of data preprocessing and cleaning for model performance.
- The significance of feature engineering in recommendation systems.
- Model selection and parameter tuning can significantly impact recommendation quality.
- Deployment and containerization make it easier to share and scale recommendation systems.
- Different architecture approaches to deploying a recommendation system

### Automation
To improve the project, consider automating various aspects:
- Use Prefect for workflow automation, scheduling tasks like `preprocess.py` and `train.py` to run at specific intervals.
- Connect to a database to retrieve and store user interactions on a daily basis, enabling real-time recommendations.

### Other Potential Improvements
- Implement user feedback loops to continuously improve recommendations based on user interactions.
- Experiment with different recommendation algorithms and ensembles to enhance recommendation quality.
- Incorporate natural language processing techniques for better understanding of user reviews.
- Optimize model training and recommendation generation for better performance and scalability.

## References
The project was inspired by and references the following articles and sites:
- [Learning to Rank for Product Recommendations](https://towardsdatascience.com/learning-to-rank-for-product-recommendations-a113221ad8a7): Discussing how to use the popular XGBoost library for Learning-to-rank(LTR) problems.
- [Recommendation System_Amazon_Electronics](https://www.kaggle.com/code/shivamardeshna/recommendation-system-amazon-electronics#Popularity-Based-Recommendation): Explaining the 6 types of the recommendations systems.
- [Implementing an Enterprise Recommendation System](https://towardsdatascience.com/implementing-an-enterprise-recommendation-system-89dd439db444): An end-to-end look at implementing a “real-world” content-based recommendation system.

## Contributions
Contributions to the project are welcome. Feel free to submit issues or pull requests to enhance the project's functionality, documentation, or overall performance.

## Tools Used
Some of the tools and libraries used in this project include:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pandas](https://pandas.pydata.org/)
- [Uvicorn](https://www.uvicorn.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [NumPy](https://numpy.org/)
- [MLflow](https://mlflow.org/)
- [Prefect](https://www.prefect.io/)
- [Hyperopt](https://hyperopt.github.io/hyperopt/)

Feel free to reach out to the project maintainers if you have any questions or need further assistance.
