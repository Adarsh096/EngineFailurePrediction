from huggingface_hub import HfApi, create_repo
import os

#common constants:
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')
HUGGINGFACE_SPACE_NAME = os.getenv('HUGGINGFACE_SPACE_NAME')
repo_id = f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_SPACE_NAME}"
HUGGINGFACE_MODEL_NAME = os.getenv('HUGGINGFACE_MODEL_NAME')
api = HfApi(token=os.getenv("HF_TOKEN"))

# Try to create the repo if it doesn't exist
try:
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", private=False)
    print(f"Space created at: {repo_id}")
except Exception as e:
    print(f"Space already exists or encountered an error: {e}")
# SYNC VARIABLES
# This sends the GitHub vars into the HF Space Environment
api.add_space_variable(repo_id=repo_id, key="HUGGINGFACE_USER_NAME", value=HUGGINGFACE_USER_NAME)
api.add_space_variable(repo_id=repo_id, key="HUGGINGFACE_MODEL_NAME", value=HUGGINGFACE_MODEL_NAME)

#upload the streamlit app files:
api.upload_folder(
    folder_path="engine_failure_prediction_project/deployment",
    repo_id=repo_id,  # the target repo
    repo_type="space", # dataset, model, or space
    path_in_repo="", # optional: subfolder path inside the repo
)
