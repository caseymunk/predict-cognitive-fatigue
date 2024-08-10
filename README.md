# cognitive-fatigue
machine learning models to predict cognitive fatigue

## Background
This repository was created as part of a research effort to learn more about cognitive fatigue and how to predict cognitive fatigue. 

## Objective
This repository holds Jupyter notebooks documenting the cognitive fatigue model development process. It also contains the final cognitive fatigue model, which can be used by running [`predict_cf.py`](/src/predict_cf.py). The smooth-pursuit eye-tracking dataset used to develop the model can be found [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6IZ0TK). 

## Model Overview
This XGBoost model uses both subjective and objective measures as inputs. See the [README](/database/README.md) in [/database](/database) for additional feature details.

### Final XGBoost Model Performance
- `Accuracy`: 0.9350
- `Precision`: 0.9440
- `Recall`: 0.9350
- `F1 Score`: 0.9334
- `ROC AUC`: 0.9238

### Hyperparameters Used
- `subsample`: 0.5
- `n_estimators`: 300 
- `min_child_weight`: 1 
- `max_depth`: 6 
- `learning_rate`: 0.1
- `gamma`: 0
- `colsample_bytree`: 0.9

### Computational Execution Details
For a machine with:
- **OS**: Windows 10 Pro
- **CPU**: 13th Gen Intel(R) Core(TM) i5-1345U (10 cores, 1.60 GHz)
- **Memory**: 16 GB of RAM

The [`predict_cf.py`](/src/predict_cf.py) script runs with:
- **Execution Time**: Approximately 0.848 seconds (on a system with the above hardware specifications)
- **Memory Usage**: Approximately 127.5 MB
- **CPU Usage**: Peaks at 11.9% during execution

## Development
Please follow the installation instructions below. 

### Prerequisites
First, clone this repository:
```bash
git clone https://github.boozallencsn.com/Munk-Casey/cognitive-fatigue.git
cd cognitive-fatigue
```

### Anaconda Installation
1. Navigate to the Anaconda IDE and create a new Python virtual environment. Ensure it is Python version 3.10, and don't install any additional packages.

Alternatively, if you prefer the command line:
```bash
conda create --name cf-venv python=3.10 # where cf-venv is the name of your virtual environment
```

2. Activate your new virtual environment and install the dependencies by running:
```bash
conda activate cf-venv # where cf-venv is the name of your virtual environment
pip install -r requirements.txt # install dependencies
```


### Non-Anaconda Installation
If you don't use conda, make sure Python 3.10 is downloaded to your computer, then open the command prompt and enter the following:
```bash
cd cognitive-fatigue
python --version # make sure it's 3.10
python -m venv cf-venv # where cf-venv is the name of your virtual environment cf = cognitive fatigue
cf-venv\Scripts\activate  # windows
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=cf-venv
```

You should now have the following tree structure within your cognitive-fatigue directory:
```
│   .gitignore
│   README.md
│   requirements.txt
│   
├───database
│       Data_eyeMovements_marginal.xlsx      
│       Data_gazeDeviation_completeTrial.xlsx
│       Data_performance.xlsx
│       Data_selfReported.xlsx
│       demo_data.csv
│       merged_df.csv
│       README.md
│       
├───models
│       cf_xgboost_pipeline.pkl
│       
├───notebooks
│       01-cmm-eda.ipynb
│       02-cmm-model-dev.ipynb
│       03-cmm-deploy.ipynb
│
├───outputs
│       predictions_20240705_124518.csv
│
└───src
        predict_cf.py
```
        NOTE: This tree structure must be maintained in order for the model to run correctly.

## Running the Model
First activate your cognitive fatigue virtual environment (cf-venv)
```bash
conda activate cf-venv # for conda installation
# or
cf-venv\Scripts\activate # for non-conda installation
```
Next, enter the project directory.
```bash
cd cognitive-fatigue
```
Note, the prediction cognitive fatigue script is located within the [/src](/src) directory, so you can run the script by typing:
```bash
python3 src/predict_cf.py <path-to-input-data>
```
path-to-input-data is the file path to the new smooth-pursuit eye tracking data for which you'd like to generate predictions. Ensure that the file path has quotes around it, as it should be a string. There is example [demo data](/database/demo_data.csv) stored in the [/database](/database) directory. See an example of how to run predictions for this demo data below:
```bash
python3 src/predict_cf.py "../database/demo_data.csv"
```

 When you run the line above, the final predictions (CSV), with a timestamp in the file name, will be saved to [/outputs](/outputs)! If you have trouble running the script with the `python3` command, try `python` instead.

## Additional Usage
This repository can also be used to recreate the model, by use of the Jupyter Notebooks located in [/notebooks](/notebooks/). There are a few intended uses for this code, listed below.

### 1. Understanding the Data
To understand the dataset, navigate to [01-cmm-eda.ipynb](/notebooks/01-cmm-eda.ipynb), where I walk through some exploratory data analysis and begin preliminary model development. This notebook walks through my thought process as the model was built and contains everything needed to replicate the model again-- reproducibility is key! If this is your first time viewing the dataset, feel free to interact with the cells to create your own visualizations and further understand the dataset. If you install any new packages, just be sure to 
```bash
pip freeze > requirements.txt
```
or if you have a mixture of conda and pip installs:
```bash
pip list --format=freeze > requirements.txt
```

### 2. Replicating Model Development
To rebuild the model, navigate to [02-cmm-model-dev.ipynb](/notebooks/02-cmm-model-dev.ipynb). This is where the XGBoost development pipeline lives. In this notebook, we go through preprocessing, feature engineering, and nested cross-validation to iterate through different model versions. This is what you would want to edit if you want to try using a different model or perform additional model tuning and view the dynamic outputs.

### 3. Using the Final Model
This was outlined in the `Running the Model` section above. There are a few assumptions that the input data must follow in order for the model to yield the expected accuracy. See below.

## Input Data Assumptions
The [raw data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6IZ0TK) from the smooth-pursuit eye-tracking study was spread across four different Excel files. This data was organized in a way that suited the study's specific analyses. In our case, I have assumed that the input data are compiled into a single table or DataFrame. 

In a real-world scenario, we would likely store real-time information in some sort of database. I assume the data would be dumped into a data lake and preorganized into a PostgreSQL database. This would eliminate duplicated columns across multiple tables, making them queryable via foreign keys. Because of this, the input data is assumed to be pulled from a PostgreSQL database with relevant columns already queried. This query can be easily set up within a function and added as a preprocessing step in the predict_cf.py script.

## Authors
This repository was written and maintained by Casey Munk in June 2024 under direction of Maggie Corry and Jared Feldman.