from huggingface_hub import HfApi

# Your Hugging Face API token
api_token = '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'

# Initialize the HfApi object
api = HfApi(token=api_token)

# Function to create a repository
def create_repo(repo_name):
    try:
        api.create_repo(repo_name, private=False)
        print(f'Repo {repo_name} created successfully.')
    except Exception as e:
        print(f'Failed to create repo {repo_name}: {e}')

# Create 100 repos
for i in range(1, 101):
    repo_name = f'testing-repo-{i}'
    create_repo(repo_name)
