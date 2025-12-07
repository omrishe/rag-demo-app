import os
import chardet
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def retrieve_file(dataset_name="Software Questions.csv"):
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", dataset_name)  # path to the dataset
    #check valid path
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at: {dataset_path}")
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        with open(dataset_path, "rb") as f:
            result = chardet.detect(f.read())
        encoding = result["encoding"]
        confidence = result["confidence"]
        logger.info(f"Detected encoding: {encoding} with confidence: {confidence}")

        # read the dataset
        df = pd.read_csv(dataset_path, encoding=encoding)  # read the dataset into a pandas dataframe
        logger.info(f"Successfully loaded dataset from {dataset_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error reading the dataset: {e}", exc_info=True)
        logger.error(f"Error encoding: {encoding}")
        logger.error(f"Error detecting encoding confidence: {confidence}")
        raise RuntimeError(f"Failed to load dataset: {e}")
