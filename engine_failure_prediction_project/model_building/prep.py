# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

#common constants:
HUGGINGFACE_DATASET_NAME = os.getenv('HUGGINGFACE_DATASET_NAME')
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = f"hf://datasets/{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_DATASET_NAME}/engine_data.csv"
engine_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# removing spaces from column names and making all column names in lowercase:
engine_dataset.columns = engine_dataset.columns.str.strip().str.replace(' ', '_').str.lower()
print(f"Transformed columns: {engine_dataset.columns.tolist()}")
#removing the duplicated rows
engine_dataset.drop_duplicates(inplace=True)

# Define the target variable for the classification task
target = 'engine_condition'

# List of numerical features in the dataset
numeric_features = [
  'engine_rpm',
  'lub_oil_pressure',
  'fuel_pressure',
  'coolant_pressure',
  'lub_oil_temp',
  'coolant_temp'
  ]

# List of categorical features in the dataset
categorical_features = []

# Define predictor matrix (X) using selected numeric and categorical features
X = engine_dataset[numeric_features + categorical_features]

# Define target variable
y = engine_dataset[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y         # To ensure class proportions match after split
)

# fitting standard scalar on train set features
scaler = StandardScaler()
scaler.fit(Xtrain)
# applying the scalar to train and test data:
Xtrain_scaled = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns)
Xtest_scaled = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns)

Xtrain_scaled.to_csv("Xtrain.csv",index=False)
Xtest_scaled.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_DATASET_NAME}",
        repo_type="dataset",
    )
