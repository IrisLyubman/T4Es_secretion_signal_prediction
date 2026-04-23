import torch
import os
import re
import pandas as pd
import argparse
from Bio import SeqIO
import joblib
import logging
import time
from tqdm import tqdm
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel


PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRETRAINED_MODEL_DISK_DIR = f"{PROJECT_ROOT_DIR}/models/prot_t5_xl_uniref50_01_08_2024/"
PRETRAINED_MODEL_HUGGINGFACE_NAME = 'Rostlab/prot_t5_xl_uniref50'
TRAINED_HEAD_DIR = f"{PROJECT_ROOT_DIR}/final_results/trained_pt5_head/model.pkl"


def get_pt5_embeddings(sequences, load_llm_from_disk, batch_size=4):
    pretrained_model_path = PRETRAINED_MODEL_DISK_DIR if load_llm_from_disk else PRETRAINED_MODEL_HUGGINGFACE_NAME
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(pretrained_model_path)
    model.eval()

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    tokenized_sequences = tokenizer(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(tokenized_sequences['input_ids'])
    attention_mask = torch.tensor(tokenized_sequences['attention_mask'])

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            sequences_representation = outputs.last_hidden_state[:, :-1, :].mean(1)
            embeddings.append(sequences_representation)

    Xs = torch.cat(embeddings)

    return Xs


def read_fasta_file(file_path):
    sequences = {}
    for seq_record in SeqIO.parse(file_path, 'fasta'):
        sequences[seq_record.id] = str(seq_record.seq)
    df = pd.DataFrame(sequences.items(), columns=['locus', 'sequence'])
    return df


def main(args):
    start_time = time.time()

    input_fasta_file = args.input_fasta_file
    output_file = args.output_file
    input_fasta_file_name = Path(input_fasta_file).stem

    if output_file is None:
        output_file = os.path.join(os.path.dirname(input_fasta_file), f'{input_fasta_file_name}_predictions.csv')

    logger = logging.getLogger('main')
    log_path = os.path.join(os.path.dirname(output_file), f'secretion_signal_{input_fasta_file_name}_batch_{args.batch_size}_log.txt')
    file_handler = logging.FileHandler(log_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    logger.info(f'Reading sequences from {input_fasta_file}...')
    reading_sequences_start_time = time.time()
    df = read_fasta_file(input_fasta_file)
    logger.info(f'Read {len(df)} sequences, time elapsed: {time.time() - reading_sequences_start_time:.2f} seconds')

    logger.info(f'Calculating embeddings for {len(df)} sequences using model PT5, '
                f'batch size {args.batch_size} and load_llm_from_disk={args.load_llm_from_disk}...')
    calc_embeddings_start_time = time.time()

    embeddings = get_pt5_embeddings(df['sequence'], args.load_llm_from_disk, args.batch_size)
    logger.info(f'Embeddings calculated, time elapsed: {time.time() - calc_embeddings_start_time:.2f} seconds')

    logger.info(f'Predicting probabilities using the trained classifier...')
    predict_probabilities_start_time = time.time()
    model = joblib.load(TRAINED_HEAD_DIR)
    predictions = model.predict_proba(embeddings)
    logger.info(f'Predictions done, time elapsed: {time.time() - predict_probabilities_start_time:.2f} seconds')

    df['T4_signal'] = predictions[:, 1]
    df.drop(columns=['sequence'], inplace=True)

    df.to_csv(output_file, index=False)

    logger.info(f'Predictions saved to {output_file}')
    logger.info(f'Elapsed time: {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta_file', help='path to a fasta file', required=True)
    parser.add_argument('--output_file', help='path to the output file with predictions. By default, the output file '
                                              'will be saved in the same directory as the input file.', required=False)
    parser.add_argument('--batch_size', help='The batch size to use for the prediction', type=int, default=16)
    parser.add_argument('--load_llm_from_disk', help='Whether to load the llm from disk, or from hugging face.'
                                                     'If load_llm_from_disk is True, the models are assumed to be in '
                                                     'T4Es_secretion_signal_prediction/models/ directory.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
