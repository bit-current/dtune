import os
import torch
import hashlib
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from huggingface_hub import HfApi, Repository, HfFileSystem, hf_hub_download, scan_cache_dir
import shutil
import subprocess
import warnings

# Option 1: Ignore all warnings
warnings.filterwarnings('ignore')

load_dotenv()

class HFManager:
    """
    Manages interactions with the Hugging Face Hub for operations such as cloning, pushing and pulling models or weights/gradients.
    """

    def __init__(
        self,
        local_dir=".",#gradients local
        hf_token=os.getenv("HF_TOKEN"),
        gradient_repo_id=None,#gradients HF
        averaged_model_repo_id=None,#averaged HF
        gradient_repo_local=None,#averaged local
        averaged_model_repo_local=None,
        averaged_miner_assignment_repo_id = None,
        averaged_miner_assignment_repo_local = None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
    
        # Initializes the HFManager with the necessary repository and authentication details.
        self.gradient_repo_id = gradient_repo_id
        self.averaged_model_repo_id = averaged_model_repo_id
        self.hf_token = hf_token
        self.device = device
        self.gradient_repo_local = gradient_repo_local
        self.averaged_model_repo_local = averaged_model_repo_local
        self.averaged_miner_assignment_repo_local = averaged_miner_assignment_repo_local
        self.averaged_miner_assignment_repo_id = averaged_miner_assignment_repo_id
        #self.local_dir = local_dir
        
        self.latest_commit_sha = {}

        # Define the local directory structure based on repository IDs but only do clone personal repo if miner
        if (self.gradient_repo_local is None) and (self.gradient_repo_id is not None):
            self.gradient_repo_local = self.gradient_repo_id.split("/")[-1]
            self.gradient_repo_local = os.path.join(local_dir, self.gradient_repo_local)

        
        if self.gradient_repo_id is not None:
            if not os.path.exists(self.gradient_repo_local): 
                os.makedirs(self.gradient_repo_local)
                
        if (self.gradient_repo_local is not None) and (self.gradient_repo_id is not None):
            self.gradient_repo = Repository(
                local_dir=self.gradient_repo_local,
                clone_from=self.gradient_repo_id,
                use_auth_token=self.hf_token,
            )
            
        if self.averaged_model_repo_local is None:
            self.averaged_model_repo_local = averaged_model_repo_id.split("/")[-1]
            self.averaged_model_repo_local = os.path.join(local_dir, self.averaged_model_repo_local)

        if not os.path.exists(self.averaged_model_repo_local):
            os.makedirs(self.averaged_model_repo_local)

        if self.averaged_model_repo_local is not None:
            self.averaged_model_repo = Repository(
                local_dir=self.averaged_model_repo_local,
                clone_from=averaged_model_repo_id,
                use_auth_token=hf_token,
            )

        if (self.averaged_miner_assignment_repo_local is None) and (self.averaged_miner_assignment_repo_id is not None):
            self.averaged_miner_assignment_repo_local = averaged_miner_assignment_repo_id.split("/")[-1]
            self.averaged_miner_assignment_repo_local = os.path.join(local_dir, self.averaged_miner_assignment_repo_local)

        if self.averaged_miner_assignment_repo_local is not None:
            if not os.path.exists(self.averaged_miner_assignment_repo_local):
                os.makedirs(self.averaged_miner_assignment_repo_local)

        if self.averaged_miner_assignment_repo_local is not None:
            self.averaged_miner_assignment_repo = Repository(
                local_dir=self.averaged_miner_assignment_repo_local,
                clone_from=averaged_miner_assignment_repo_id,
                use_auth_token=hf_token,
            )

        self.api = HfApi()
        # Get the latest commit SHA for synchronization checks
        self.latest_model_commit_sha = self.get_latest_commit_sha(self.gradient_repo_id)

        self.check_git_lfs()

        # Initialize Git LFS for each repository
        self.init_git_lfs(self.gradient_repo_local)
        self.init_git_lfs(self.averaged_model_repo_local)
        self.init_git_lfs(self.averaged_miner_assignment_repo_local)

    def check_git_lfs(self):
        try:
            subprocess.run(["git", "lfs", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise RuntimeError("Git LFS is not installed. Please install it: https://git-lfs.github.com/")

    def init_git_lfs(self, repo_path):
        if repo_path:
            subprocess.run(["git", "lfs", "install"], cwd=repo_path, check=True)
            subprocess.run(["git", "lfs", "track", "*.pt"], cwd=repo_path, check=True)
            subprocess.run(["git", "add", ".gitattributes"], cwd=repo_path, check=True)

    @staticmethod
    def clear_hf_cache():
        # Get the cache directory
        hf_cache_info = scan_cache_dir()
        commit_hashes = [
            revision.commit_hash
            for repo in hf_cache_info.repos
            for revision in repo.revisions
        ]

        # Check if the cache directory exists
        delete_strategy = scan_cache_dir().delete_revisions(*commit_hashes)

        print("Will free " + delete_strategy.expected_freed_size_str)
        delete_strategy.execute()

    @staticmethod
    def git_prune_and_refresh(repo_path):
        """
        Change to the specified repository directory, execute 'git lfs prune', and revert to the original directory.
        """
        original_dir = os.getcwd()
        try:
            os.chdir(repo_path)
            subprocess.run(['git', 'config', 'pull.rebase', 'true'], check=True)   
            subprocess.run(['git', 'pull', '--force'], check=True)
            subprocess.run(['git', 'lfs', 'prune'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to prune Git LFS objects: {e}")
        finally:
            os.chdir(original_dir)

    def git_push_force(self, repo):
        """
        Executes a git push --force command using subprocess, setting the upstream branch if necessary.
        """
        try:
            command = ["git", "push", "--set-upstream", "origin", "main", "--force"]
            process = subprocess.run(
                command,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
                cwd=repo.local_dir,
            )
            print(f"Force push successful: {process.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Force push failed: {e.stderr}")
            if "remote: error: GH001" in e.stderr:
                print("The repository might have Git LFS disabled. Please enable it in the repository settings.")
        except Exception as e:
            print(f"An error occurred during force push: {str(e)}")


    def push_gradients(self, files_to_send):
        """
        Stages, commits, squashes, and pushes changes to the configured repository.
        Also prunes unnecessary Git LFS objects to free up storage.
        """
        try:
            # Stage the changes
            if type(files_to_send) == str:
                files_to_send = [files_to_send]
            
            for file_to_send in files_to_send:
                subprocess.run(["git", "lfs", "track", file_to_send], cwd=self.gradient_repo_local, check=True)
                self.gradient_repo.git_add(file_to_send)            
            
            # Commit with a unified message
            self.gradient_repo.git_commit("Squashed commits - update model gradients")
            
            # Push the changes to the repository
            self.git_push_force(self.gradient_repo)

            self.api.super_squash_history(repo_id=self.gradient_repo_id)
            
            # Prune unneeded Git LFS objects and pull the squashed version locally
            self.git_prune_and_refresh(self.gradient_repo_local)  # Clean up unused LFS objects       
            
        except Exception as e:
            print(f"Failed to push changes: {e}")

    def push_averaged_model(self, path_to_model, commit_message="Pushing model to Hub"):
        try:
            # Stage the changes
            if type(path_to_model) == str:
                path_to_model = [path_to_model]
                
            for path_to_add in path_to_model:
                subprocess.run(["git", "lfs", "track", path_to_add], cwd=self.averaged_model_repo_local, check=True)
                self.averaged_model_repo.git_add(path_to_add)
            
            # Squash commits into a single one before pushing
            
            # Commit with a unified message
            self.averaged_model_repo.git_commit("Squashed commits - update model gradients")
            
            self.git_push_force(self.averaged_model_repo)

            self.api.super_squash_history(repo_id=self.averaged_model_repo_id)

            # Prune unneeded Git LFS objects and pull the squashed version locally
            self.git_prune_and_refresh(self.averaged_model_repo_local)
            
            # Push the changes to the repository
            
        except Exception as e:
            print(f"Failed to push changes: {e}")

    def push_miner_assignemnts(self, path_to_assignment, commit_message="Pushing model to Hub"):
        try:
            # Stage the changes
            if type(path_to_assignment) == str:
                path_to_assignment = [path_to_assignment]

            for path_to_add in path_to_assignment:
                subprocess.run(["git", "lfs", "track", path], cwd=self.averaged_miner_assignment_repo_local, check=True)
                self.averaged_miner_assignment_repo.git_add(path_to_add)
            
            # Squash commits into a single one before pushing
            
            # Commit with a unified message
            self.averaged_miner_assignment_repo.git_commit("Squashed commits - update model gradients")
            
            self.git_push_force(self.averaged_miner_assignment_repo)

            self.api.super_squash_history(repo_id=self.averaged_miner_assignment_repo_id)

            # Prune unneeded Git LFS objects and pull the squashed version locally
            self.git_prune_and_refresh(self.averaged_miner_assignment_repo_local)
            
            # Push the changes to the repository
            
        except Exception as e:
            print(f"Failed to push changes: {e}")


    def get_latest_commit_sha(self, repo):
        """
        Fetches the latest commit SHA of the specified repository from the Hugging Face Hub.
        """
        try:
            repo_info = self.api.repo_info(repo)
            latest_commit_sha = repo_info.sha
            # print(latest_commit_sha)
            return latest_commit_sha
        except Exception as e:
            print(f"Failed to fetch latest commit SHA: {e}")
            return None

    def check_for_new_submissions(self, repo): #FIXME check we're not using this double FIXME make it use a dict
        ## Make valis check for new assignments. To start a new vali cycle
        """
        Compares the current commit SHA with the latest to determine if there are new submissions.
        """
        try:
            repos_latest_commit = self.latest_commit_sha[repo]
        except KeyError:
            repos_latest_commit = None

        current_commit_sha = self.get_latest_commit_sha(repo)
        if current_commit_sha != repos_latest_commit:
            self.latest_commit_sha[repo] = current_commit_sha
            return True
        return False

    def update_model(self, model, model_file_name="averaged_model.pt"):
        """
        Loads an updated model from a .pt file and updates the in-memory model's parameters.
        """
        try:
            model_path = os.path.join(self.averaged_model_repo_local, model_file_name)
            if os.path.exists(model_path):
                new_state_dict = torch.load(model_path, map_location=self.device)
                temp_state_dict = model.state_dict()
                temp_state_dict.update(new_state_dict)
                # for name, param in model.named_parameters():
                #     if name in model_state_dict:
                #         param.data.copy_(model_state_dict[name])
                model.load_state_dict(temp_state_dict)
                model.train()
                print(f"Model updated from local path: {model_path}")
                return model
            else:
                raise FileNotFoundError(f"{model_file_name} not found in the repository.")
        except FileNotFoundError as e:
            print(f"Failure to update model: {e}")
        except Exception as er:
            print("Attempting to load corrupt/wrong weights")

    def get_local_gradient_directory(self):
        """Return the local directory of the repository."""
        return self.gradient_repo_local

    def get_averaged_model_directory(self):
        """Return the local directory of the repository."""
        return self.averaged_model_repo_local

    def get_averaged_miner_assignment_directory(self):
        """Return the local directory of the repository."""
        return self.averaged_miner_assignment_repo_local

    def pull_latest_model(self):
        try:
            shutil.rmtree(self.averaged_model_repo_local)
            print(f"Successfully removed directory: {self.averaged_model_repo_local}")
        except FileNotFoundError:
            print(f"Directory not found: {self.averaged_model_repo_local}")
        except Exception as e:
            print(f"An error occurred while removing the directory: {str(e)}")
        self.averaged_model_repo = Repository(
                local_dir=self.averaged_model_repo_local,
                clone_from=self.averaged_model_repo_id,
                use_auth_token=self.hf_token,
            )

    def pull_latest_assignments(self):
        try:
            shutil.rmtree(self.averaged_miner_assignment_repo_local)
            print(f"Successfully removed directory: {self.averaged_miner_assignment_repo_local}")
        except FileNotFoundError:
            print(f"Directory not found: {self.averaged_miner_assignment_repo_local}")
        except Exception as e:
            print(f"An error occurred while removing the directory: {str(e)}")
        self.averaged_miner_assignment_repo = Repository(
                local_dir=self.averaged_miner_assignment_repo_local,
                clone_from=self.averaged_miner_assignment_repo_id,
                use_auth_token=self.hf_token,
            )

    def receive_gradients(self, miner_repo_id, weights_file_name="gradients.pt", path_only=False):
        try: #TODO Add some garbage collection.
                       # Construct the file path in the HfFileSystem
            file_path = f"{miner_repo_id}/{weights_file_name}"

            # Check if the file exists
            if not self.fs.exists(file_path):
                print(f"No file '{weights_file_name}' found in repo '{miner_repo_id}'.")
                return None

            # Get file metadata
            info = self.fs.info(file_path)

            # Extract the last commit date
            last_commit_info = info.get('last_commit')
            if not last_commit_info:
                print(f"No commit information available for '{weights_file_name}' in repo '{miner_repo_id}'.")
                return None

            last_modified = last_commit_info.date  # This is a datetime object
            if not isinstance(last_modified, datetime):
                print(f"Invalid date format for last commit of '{weights_file_name}' in repo '{miner_repo_id}'.")
                return None
            # Current UTC time
            now = datetime.now(timezone.utc)

            # Current UTC time
            now = datetime.now(timezone.utc)

            # Check if the file was modified within the last 24 hours
            if now - last_modified > timedelta(hours=24):
                print(
                    f"Gradients file '{weights_file_name}' in repo '{miner_repo_id}' was not updated in the last 24 hours. Skipping"
                )
                return None
            # Download the gradients file from Hugging Face Hub
            weights_file_path = hf_hub_download(
                repo_id=miner_repo_id, filename=weights_file_name, use_auth_token=True
            )
            # Load the gradients directly using torch.load
            
            if path_only:
                return weights_file_path
            else:
                miner_weights = torch.load(weights_file_path, map_location=self.device)
                os.remove(weights_file_path)
                return miner_weights
        
        except Exception as e:
            print(f"Error receiving gradients from Hugging Face: {e}")

    def clean_repo(self, repo_name):
        """
        Deletes all files in the specified repository, squashes its history,
        and prepares it for a fresh start.
        
        :param repo_name: The name of the repository to clean ('gradient', 'averaged_model', or 'averaged_miner_assignment')
        """
        if repo_name == 'gradient':
            repo = self.gradient_repo
            local_dir = self.gradient_repo_local
        elif repo_name == 'averaged_model':
            repo = self.averaged_model_repo
            local_dir = self.averaged_model_repo_local
        elif repo_name == 'averaged_miner_assignment':
            repo = self.averaged_miner_assignment_repo
            local_dir = self.averaged_miner_assignment_repo_local
        else:
            raise ValueError("Invalid repo_name. Choose 'gradient', 'averaged_model', or 'averaged_miner_assignment'")

        try:
            # Step 1: Ensure we're on the main branch
            subprocess.run(["git", "checkout", "main"], cwd=local_dir, check=True)

            # Step 2: Remove all files except .git
            for item in os.listdir(local_dir):
                item_path = os.path.join(local_dir, item)
                if item != '.git':
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)

            # Step 3: Stage all changes
            subprocess.run(["git", "add", "-A"], cwd=local_dir, check=True)

            # Step 4: Check if there are changes to commit
            status = subprocess.run(["git", "status", "--porcelain"], cwd=local_dir, capture_output=True, text=True)
            if status.stdout.strip():
                # There are changes to commit
                subprocess.run(["git", "commit", "-m", "Remove all files for fresh start"], cwd=local_dir, check=True)

            # Step 5: Create a new orphan branch
            new_branch = "fresh_start"
            subprocess.run(["git", "checkout", "--orphan", new_branch], cwd=local_dir, check=True)

            # Step 6: Remove all files from the new branch (should be empty already, but just in case)
            for item in os.listdir(local_dir):
                item_path = os.path.join(local_dir, item)
                if item != '.git':
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)

            # Step 7: Create an empty .gitkeep file and commit it
            open(os.path.join(local_dir, ".gitkeep"), "w").close()
            subprocess.run(["git", "add", ".gitkeep"], cwd=local_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit for fresh start"], cwd=local_dir, check=True)

            # Step 8: Rename the new branch to main
            subprocess.run(["git", "branch", "-D", "main"], cwd=local_dir, check=True)
            subprocess.run(["git", "branch", "-m", new_branch, "main"], cwd=local_dir, check=True)

            # Step 9: Force push to remote
            self.git_push_force(repo)

            # Step 10: Clean up
            subprocess.run(["git", "gc", "--aggressive", "--prune=all"], cwd=local_dir, check=True)

            print(f"Repository {repo_name} has been cleaned and its history has been reset.")

        except Exception as e:
            print(f"An error occurred while cleaning the repository: {str(e)}")
            raise



    


class LocalHFManager:
    def __init__(self, my_repo_id="local_models"):
        self.my_repo_id = my_repo_id
        # Ensure the local directory exists
        os.makedirs(self.my_repo_id, exist_ok=True)
        self.model_hash_file = os.path.join(self.my_repo_id, "model_hash.txt")
        # Initialize model hash value
        self.last_known_hash = None

    def set_model_hash(self, hash_value):
        """Sets and saves the latest model hash to the hash file."""
        with open(self.model_hash_file, "w") as file:
            file.write(hash_value)
        print(f"Set latest model hash to: {hash_value}")

    def check_for_new_submissions(self):
        """Checks if a new or updated model is available."""
        model_file_path = os.path.join(self.my_repo_id, "averaged_model.pt")
        if not os.path.exists(model_file_path):
            print("No model available.")
            return False

        with open(model_file_path, "rb") as file:
            file_hash = hashlib.sha256(file.read()).hexdigest()

        if self.last_known_hash is None or self.last_known_hash != file_hash:
            print("New or updated model found. Updating model...")
            self.last_known_hash = file_hash
            return True
        return False

    def update_model(self, model):
        """Updates an existing model's state dict from a .pt file."""
        model_file_path = os.path.join(self.my_repo_id, "averaged_model.pt")
        if os.path.exists(model_file_path):
            averaged_state_dict = torch.load(model_file_path)
            model_state_dict = model.state_dict()
            model_state_dict.update(averaged_state_dict)
            model.load_state_dict(model_state_dict)

            model.train()  # Or model.eval(), depending on your use case
            return model
            print(f"Model updated from local path: {model_file_path}")
        else:
            print(f"Model file not found: {model_file_path}")

def model_hash(state_dict):
    # Convert the state_dict to CPU to ensure consistency
    cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}

    # Serialize the state_dict to a string
    serialized_state_dict = str({k: v.numpy().tolist() for k, v in cpu_state_dict.items()})

    # Create a hash from the serialized state_dict
    hash_object = hashlib.md5(serialized_state_dict.encode())
    model_hash = hash_object.hexdigest()
    
    return model_hash
