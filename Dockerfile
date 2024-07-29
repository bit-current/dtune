# syntax=docker/dockerfile:experimental

# Base stage: installs all dependencies
FROM python:3.10.12 AS base

WORKDIR /RDbara

# Copy only requirements file to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Install additional packages
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install

RUN pip install bittensor huggingface_hub

# Intermediate stage: installs your package
FROM base AS builder

WORKDIR /RDbara

# Copy application code
COPY . .

#RUN git checkout hierarchical_validator_arch

# Final stage: contains the application code and the script
FROM python:3.10.12 AS final

WORKDIR /RDbara

# Copy installed dependencies from the base stage
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy the application code from the builder stage
COPY --from=builder /RDbara /RDbara

# Copy the run_additional_commands.sh script
COPY run_additional_commands.sh /RDbara/run_additional_commands.sh

RUN chmod +x /RDbara/run_additional_commands.sh

CMD env >> /etc/environment;

ENTRYPOINT ["/bin/bash", "/RDbara/run_additional_commands.sh"]