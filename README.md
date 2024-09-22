# Neurosymbolic Mental Health Monitoring during COVID-19

## Overview
This repository contains the source code and supplementary resources for our paper titled "A Domain-Agnostic Neurosymbolic Approach for Big Social Data Analysis: Evaluating Mental Health Sentiment on Social Media During COVID-19." This study introduces a neurosymbolic AI method to analyze mental health sentiment on platforms such as Twitter and Reddit during the COVID-19 pandemic. The paper has been submitted to IEEE Big Data.

## Repository Contents
- `utils/`: Contains utility scripts for various tasks:
  - `data_processing.py`: General data processing functions.
  - `initial_data_setup.py`: Prepares initial data configurations.
  - `sedo_weight_calculation.py`: Calculates SEDO weights for model training.
  - `text_cleaning.py`: Scripts for cleaning and preparing text data.
  - `train_test_split.py`: Splits data into training and testing datasets.
  - `tweet_filters.py`: Filters tweets based on specific criteria.
  - `word_embedding_preparation.py`: Prepares word embeddings for analysis.
  - `execute_*.py`: Scripts to execute specific tasks like classifier training, data processing, etc.
- `models/`: Directory to store pre-trained models and configurations (To be released later upon acceptance of the paper).
- `data/`: Directory to store datasets used in the analysis (Complete dataset will be released upon acceptance of the paper).
- `machine_learning_model_training.py`: Script for training machine learning models.
- `nlp_model_training.py`: Script for training NLP models.
- `process_tweets.py`: Processes tweets for analysis.
- `topic_model_training.py`: Script for training topic models.
- `LICENSE`: The project's license file.
- `README.md`: This file, providing an overview and instructions.
- `requirements.txt`: Lists all dependencies required to run the project.

## Data Sharing
Access to the datasets used in our study is currently restricted to tweet IDs due to Twitter's data sharing policies. Researchers can access these IDs and related datasets via the following link: [Access Tweet IDs and Datasets](https://example.com/dataset-link). These IDs can be hydrated using Twitter API tools for replication of our study or further analysis.

## Installation
To set up your environment for replicating our analysis or adapting the methodologies:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repository-link.git
   cd your-repository-directory
2. **Install Dependencies** Ensure you have Python 3.x installed, and then install the required dependencies by running:
    ```bash
    pip install -r requirements.txt

## Usage
To use the scripts for processing data, training models, and performing analysis:

1. **Data Preparation** Navigate to the utils/ directory and run the data preparation scripts to set up the data for analysis.
   ```bash
   python utils/execute_data_processing.py
   python utils/execute_classifier_training.py
2. **Model Training** Execute the model training scripts to train the neurosymbolic models as described in our study.
    ```bash
    python machine_learning_model_training.py
    python nlp_model_training.py

## Contributing
We welcome contributions to improve the project. Please fork the repository, commit your changes to a new branch, and submit a pull request for review.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact
For any queries related to this repository or the research paper, please contact:

- Vedant Khandelwal (vedant@email.sc.edu)