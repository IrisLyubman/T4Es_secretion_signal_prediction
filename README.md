# T3Es_secretion_signal_prediction

This repository contains the code for predicting the T4Es secretion signal in bacterial proteins,
based on trained models, as described in the paper: Effectidor II.

## Requirements

Clone the repository, create a conda environment using env.yml, and activate it.

## Usage

Create a fasta file that contains the 100 N-terminal amino acids of bacterial protein sequences.
Then run the following script:

python src/inference/predict_secretion_signal.py --input_fasta_file <path_to_fasta_file>

## Advanced Usage

The repository also contains all the source code used to train the models and compare the results, separated to multiple scripts.
You're welcome to use them for your own research.