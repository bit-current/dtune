import csv
import re
import subprocess
from io import StringIO
import pandas as pd

# Vast.ai show instances command
show_instances_command = "vastai show instances"

# Function to run the vastai show instances command and get the output in memory
def run_show_instances_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    return result.stdout

def process_tsv_data(tsv_data):
    tsv_data = re.sub(r'[^\S\n]+', ' ', tsv_data)
    tsv_data = tsv_data.replace(" \n", "\n").replace(" ", ",")
    return tsv_data

# Read the TSV data and collect all instance IDs
def get_instance_ids(tsv_data):
    processed_tsv_data = process_tsv_data(tsv_data)
    df = pd.read_csv(StringIO(processed_tsv_data), delimiter=',')
    instance_ids = df['ID'].tolist()
    return instance_ids

# Destroy instances using vast.ai CLI
def destroy_instances(instance_ids):
    if instance_ids:
        instance_ids = [str(instance_id) for instance_id in instance_ids]
        print(f'Attempting to destroy: {" ".join(instance_ids)}')
        command = f'vastai destroy instances {" ".join(instance_ids)}'
        subprocess.run(command, shell=True)
    else:
        print("No instances to destroy.")

# Main function
def main():
    # Run the show instances command and get the TSV data
    tsv_data = run_show_instances_command(show_instances_command)
    
    # Get all instance IDs
    instance_ids = get_instance_ids(tsv_data)
    
    # Destroy instances
    destroy_instances(instance_ids)

if __name__ == '__main__':
    main()