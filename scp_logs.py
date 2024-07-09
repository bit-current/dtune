import subprocess
import time
import pandas as pd
import io
import os 
import time

def get_host_info():
    # Replace this with your actual command to get the stdout
    command = "vastai show instances"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout

def preprocess_stdout(stdout):
    # Split the string into lines
    lines = stdout.strip().split('\n')
    
    # Modify the header
    header = lines[0].replace("SSH Addr", "SSH_Addr").replace("SSH Port", "SSH_Port").replace("Net up","Net_up").replace("Net down","Net_down").replace("Util. %", "Util_p")
    
    # Join the lines back together
    return '\n'.join([header] + lines[1:])

def parse_host_info(stdout):
    # Preprocess the stdout
    processed_stdout = preprocess_stdout(stdout)
    
    # Convert the processed stdout to a DataFrame
    df = pd.read_csv(io.StringIO(processed_stdout), delim_whitespace=True)

    # Extract relevant information
    hosts = []
    for _, row in df.iterrows():
        hosts.append({
            "host": row['SSH_Addr'],
            "port": int(row['SSH_Port']),
            "ID":str(row["ID"])
        })
    return hosts

# Define the files to copy
remote_files = [
    "/RDbara/validator.err",
    "/RDbara/validator.log",
    "/RDbara/miner.err",
    "/RDbara/miner.log",
    "/RDbara/averager.err",
    "/RDbara/averager.log"
]

# Define the local directory to save the files
timestamp = time.time()
local_directory = f"../trash_dir/logs/{str(timestamp).split('.')[0]}"
os.makedirs(local_directory)

def scp_files(host, port):
    for remote_file in remote_files:
        local_file = f"{local_directory}/{host['ID']}_{str(host['port'])}_{host['host']}_{remote_file.split('/')[-1]}"
        command = [
            "scp",
            "-P", str(port),
            "-o", "StrictHostKeyChecking=no",
            f"root@{host['host']}:{remote_file}",
            local_file,
        ]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully copied {remote_file} from {host['host']} to {local_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to copy {remote_file} from {host['host']} to {local_file}: {e}")

def append_metagraph_log():
    command = "btcli s metagraph --subtensor.network test --netuid 100"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    
    # Append the output to metagraph.log
    log_file_path = f"{local_directory}/metagraph.log"
    with open(log_file_path, 'a') as log_file:
        log_file.write("\n")
        log_file.write(result.stdout)

def main():
    while True:
        stdout = get_host_info()
        hosts = parse_host_info(stdout)
        
        for host in hosts:
            scp_files(host, host['port'])
        
        append_metagraph_log()
        time.sleep(120)

if __name__ == "__main__":
    main()