# Final XGBoost Cognitive Fatigue model prediction generation script
# Written by Casey Munk on 7/3/2024

import pandas as pd
import pickle
import os
import sys
from datetime import datetime
from sklearn.pipeline import Pipeline
from pathlib import Path

# set up global constants -- working directories
PROJ_DIRECTORY = Path.cwd() # this for when user runs src/predict_cf.py in bash
DATA_DIRECTORY = os.path.join(PROJ_DIRECTORY, 'database')
OUTPUT_DIRECTORY = os.path.join(PROJ_DIRECTORY, 'outputs')
MODEL_DIRECTORY = os.path.join(PROJ_DIRECTORY, 'models')

def get_mfd(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the mental fatigue derivative of the marginal gaze deviation with respect to change in mental fatigue (before and after).
    
    Args:
    final_df (pd.DataFrame): Merged, partially preprocessed DataFrame containing the columns 'gazeDeviation_marginal', 'mentalFatigue', and 'mentalFatigue_before'.
    
    Returns:
    pd.DataFrame: The DataFrame with an additional 'mental_derivative' column.
    """
    final_df['mental_derivative'] = None
    
    for j in range(0, final_df.shape[0] - 1, 2):
        if pd.isna(final_df['mentalFatigue'].iloc[j]) or (final_df['mentalFatigue'].iloc[j] - final_df['mentalFatigue_before'].iloc[j]) == 0: # results in a division by 0
            # if no change in mental fatigue then just take the difference in gaze deviation
            final_df.loc[j, 'mental_derivative'] = (final_df['gazeDeviation_marginal'].iloc[j+1] - final_df['gazeDeviation_marginal'].iloc[j])
            final_df.loc[j+1, 'mental_derivative'] = (final_df['gazeDeviation_marginal'].iloc[j+1] - final_df['gazeDeviation_marginal'].iloc[j])
        else:
            # mental fatigue derivative calculation for the current and next row
            final_df.loc[j, 'mental_derivative'] = (final_df['gazeDeviation_marginal'].iloc[j+1] - final_df['gazeDeviation_marginal'].iloc[j]) / (final_df['mentalFatigue'].iloc[j] - final_df['mentalFatigue_before'].iloc[j])
            final_df.loc[j+1, 'mental_derivative'] = (final_df['gazeDeviation_marginal'].iloc[j+1] - final_df['gazeDeviation_marginal'].iloc[j]) / (final_df['mentalFatigue'].iloc[j+1] - final_df['mentalFatigue_before'].iloc[j+1])

    # error catching in case the number of rows is odd in test split
    if final_df.shape[0] % 2 != 0:
        last_idx = final_df.shape[0] - 1
        if pd.isna(final_df['mentalFatigue'].iloc[j]) or (final_df['mentalFatigue'].iloc[last_idx] - final_df['mentalFatigue_before'].iloc[last_idx]) == 0:
            # compute the difference between the last row and the second to last row
            final_df.loc[last_idx, 'mental_derivative'] = (final_df['gazeDeviation_marginal'].iloc[last_idx] - final_df['gazeDeviation_marginal'].iloc[last_idx - 1])
        else:
            final_df.loc[last_idx, 'mental_derivative'] = (final_df['gazeDeviation_marginal'].iloc[last_idx] - final_df['gazeDeviation_marginal'].iloc[last_idx - 1]) / (final_df['mentalFatigue'].iloc[last_idx] - final_df['mentalFatigue_before'].iloc[last_idx])
    
    return final_df

def load_pipeline(pipeline_path: str) -> Pipeline:
    """
    Load (deserialize) the pickled XGBoost pipeline.
    
    Args:
    pipeline_path (str): Path to the pickled pipeline file.
    
    Returns:
    Pipeline: The loaded XGBoost machine learning pipeline.
    """
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def wrangle_data(input_data_path: str) -> pd.DataFrame:
    """
    Preprocess the input data and calculate the mental fatigue derivative.
    
    Args:
    input_data_path (str): Path to the input CSV file.
    
    Returns:
    pd.DataFrame: Preprocessed data with additional 'mental_derivative' column.
    """
    data = pd.read_csv(input_data_path)
    data = data.drop(['subjectID', 'session', 'trial'], axis=1)
    data = get_mfd(data)
    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/predict_cf.py <input_data_path>")
        sys.exit(1)
    # parse path str from user input
    input_data_path = sys.argv[1]

    # get pipeline file path for this user from global constant
    pipeline = 'cf_xgboost_pipeline.pkl'
    pipeline_path = os.path.join(MODEL_DIRECTORY, pipeline)
    
    # load model, wrangle data, make predictions
    pipeline = load_pipeline(pipeline_path)
    data = wrangle_data(input_data_path)
    predictions = pipeline.predict(data)

    # generate the current timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # save predictions to a csv file in /outputs
    output_filename = f"predictions_{timestamp}.csv"
    output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_path, index=False)
    
    # notify user of final prediction location
    print(f"Success! Predictions have been saved to {output_path}")