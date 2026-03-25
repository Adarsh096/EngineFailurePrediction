from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

HUGGINGFACE_DATASET_NAME = os.getenv('HUGGINGFACE_DATASET_NAME')
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')

repo_id = f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_DATASET_NAME}"
print(f"repo_id: {repo_id}")

repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="engine_failure_prediction_project/data",#only uploading the dataset
    repo_id=repo_id,
    repo_type=repo_type,
)
