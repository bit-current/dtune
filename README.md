## Hivetrain Mission
**Hivetrain** is dedicated to developing a decentralized model training platform on the Bittensor network. Our goal is to establish a scalable system that fosters collaboration among AI experts to build advanced multimodal models. We prioritize rewarding contributors for both their computational contributions and their innovations in model advancement.

## Distributed Continual Pretraining Subnet
**Distributed Continual Pretraining** represents a cutting-edge approach within our network aimed at perpetual training and enhancement of foundational models. This initiative seeks to democratize access to large-scale open-source language models, challenging the existing monopolies held by proprietary entities.

## Project Goals
Our chief aim is to facilitate the training of a leading open-source large language model (LLM) through distributed methodologies. We are committed to democratizing access to training at the trillion-parameter scale, engaging a wide community of contributors.

## How to Participate
1. **Run a Miner**: Contribute by providing computational resources or refining the model through hyperparameter adjustments.
2. **Run a Validator**: Validate and verify the work performed by miners.
3. **Propose Improvements**: Suggest new architectures, training algorithms, or other enhancements.

## System Architecture
Our architecture is structured into three main tiers, optimized for scalable and distributed training of large language models:

### Miners
Miners are crucial, executing primary training tasks using **Weight-Decomposed Low-Rank Adaptation (DoRA)**:
- **Efficient Fine-Tuning**: Leverages only about 5% of the original model parameters for effective ongoing pretraining.
- **Distributed Processing**: Each miner processes a segment of the total training data, facilitating extensive parallelism.
- **Accessible Training**: Ensures compatibility with moderately powered GPUs, expanding our base of potential contributors.

**Process**:
1. Download training data and sync the current model state.
2. Execute DoRA-based training on a GPU.
3. Update and compute new weight matrices.
4. Upload results to a Hugging Face repository for validation.

### Validators
Validators maintain the training quality and integrity across the network by following these steps:
1. Retrieve weights and the averaged model from Hugging Face.
2. Evaluate the quality of updates against baseline metrics.
3. Assign scores and rewards based on miner contributions and performance.
4. Participate in initial weight averaging and submit results for further averaging.

### Centralized Averager
The averager plays a pivotal role in integrating validated updates from validators into a comprehensive model update:
1. Gather and average weights verified by validators.
2. Apply advanced algorithms to integrate these updates seamlessly.
3. Update the main model state and release the latest version on Hugging Face.

**Note**: Efforts are underway to decentralize the averaging process to enhance validation procedures. The averager's script is available for public review.

## Optimal Training Practices
- **Train Harder**: Increase resources by utilizing additional GPUs and devices.
- **Train Smarter**: Employ sophisticated algorithms and adapt initial scripts to yield improved performance outcomes.

## Roadmap
- Implementation of model parallelism.
- Expansion to fully multimodal capabilities.
