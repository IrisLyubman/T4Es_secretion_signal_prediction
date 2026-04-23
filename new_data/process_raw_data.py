from pathlib import Path
from Bio import SeqIO
import subprocess
import logging
import argparse
import pandas as pd
import random
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = SCRIPT_DIR / 'raw_data'

T4E_PATH = RAW_DATA_PATH / 'T4Es.faa'
NON_T4E_PATH = RAW_DATA_PATH / 'non_T4Es.faa'

LOG_MESSAGE_FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'

MMSEQS_OUTPUT_FORMAT = 'query,target,fident,qcov,tcov,bits,evalue'
MMSEQS_OUTPUT_HEADER = MMSEQS_OUTPUT_FORMAT.split(',')


def setup_logger(outputs_dir):
    logger = logging.getLogger('main')
    file_handler = logging.FileHandler(outputs_dir / 'log.txt', mode='w')
    formatter = logging.Formatter(LOG_MESSAGE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger


def run_command(logger, command):
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_message = f'Error in command: "{e.cmd}": {e.stderr}'
        logger.exception(error_message)
        raise e


def cut_C_terminal(logger, records, output_path, record_id_prefix_to_add):
    for record in records:
        record.seq = record.seq[-100:]
        record.id = f'{record_id_prefix_to_add}:{record.id}'

    SeqIO.write(records, output_path, 'fasta')
    logger.info(f'Wrote {len(records)} records to {output_path}.')


def cluster_records(logger, input_fasta_path, output_dir, output_fasta_path, min_e_value, min_coverage, min_identity):
    number_of_sequences = len(list(SeqIO.parse(input_fasta_path, 'fasta')))

    output_dir.mkdir(exist_ok=True, parents=True)
    cmd = f'mmseqs easy-cluster {input_fasta_path} {output_dir / "result"} {output_dir / "tmp_dir"} -e {min_e_value} ' \
          f'--min-seq-id {min_identity} -c {min_coverage} --threads 1'
    run_command(logger, cmd)
    shutil.copy(output_dir / 'result_rep_seq.fasta', output_fasta_path)

    number_of_representatives = len(list(SeqIO.parse(output_fasta_path, 'fasta')))
    logger.info(f'Wrote representatives of {input_fasta_path} to {output_fasta_path}. '
                f'Reduced from {number_of_sequences} to {number_of_representatives} sequences.')


def find_homologs(logger, query_fasta_path, target_fasta_path, output_fasta_path, tmp_dir_path, min_e_value,
                  min_coverage, min_identity):
    cmd = f'mmseqs easy-search {query_fasta_path} {target_fasta_path} {output_fasta_path} {tmp_dir_path} ' \
          f'-e {min_e_value} --min-seq-id {min_identity} -c {min_coverage} --threads 1 ' \
          f'--format-output {MMSEQS_OUTPUT_FORMAT}'
    run_command(logger, cmd)

    homologs_df = pd.read_csv(output_fasta_path, sep='\t', names=MMSEQS_OUTPUT_HEADER)
    homologs_df.to_csv(output_fasta_path, index=False)


def create_negative_final_dataset(logger, negative_fasta_path, t4e_fasta_path, output_dir, min_e_value, min_coverage,
                                  min_identity):
    find_homologs(logger, negative_fasta_path, t4e_fasta_path, output_dir / 'negative_homologs_t4es.m8', output_dir / 'tmp',
                  min_e_value, min_coverage, min_identity)
    negative_homologs_t4es_df = pd.read_csv(output_dir / 'negative_homologs_t4es.m8')
    negative_homologs_t4es_ids = set(negative_homologs_t4es_df['query'])

    negative_records = list(SeqIO.parse(negative_fasta_path, 'fasta'))
    negative_final_records = [record for record in negative_records if record.id not in negative_homologs_t4es_ids]
    SeqIO.write(negative_final_records, output_dir / 'negative_final.faa', 'fasta')

    logger.info(f'Out of {len(negative_records)} negative sequences, {len(negative_final_records)} were chosen as real negative, '
                f'after removing {len(negative_homologs_t4es_ids)} homologs to T4Es.')


def split_fasta(logger, input_fasta, output_train_fasta, output_test_fasta, test_ratio):
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    random.shuffle(sequences)

    split_idx = int(len(sequences) * test_ratio)
    test_seqs = sequences[:split_idx]
    train_seqs = sequences[split_idx:]

    # Write training set
    with open(output_train_fasta, "w") as train_out:
        SeqIO.write(train_seqs, train_out, "fasta")

    # Write testing set
    with open(output_test_fasta, "w") as test_out:
        SeqIO.write(test_seqs, test_out, "fasta")

    logger.info(f"Split {len(sequences)} sequences from {input_fasta} into {len(train_seqs)} train and {len(test_seqs)} test sequences.")


def unite_fasta_files(logger, fasta1, fasta2, output_fasta):
    sequences = []

    for fasta in [fasta1, fasta2]:
        sequences.extend(list(SeqIO.parse(fasta, "fasta")))

    with open(output_fasta, "w") as output:
        SeqIO.write(sequences, output, "fasta")


def verify_train_and_test_non_homologs(logger, positive_train_fasta, positive_test_fasta, negative_train_fasta,
                                       negative_test_fasta, output_dir, min_e_value, min_coverage, min_identity):
    all_train_records = output_dir / 'all_train_records.faa'
    unite_fasta_files(logger, positive_train_fasta, negative_train_fasta, all_train_records)
    all_test_records = output_dir / 'all_test_records.faa'
    unite_fasta_files(logger, positive_test_fasta, negative_test_fasta, all_test_records)

    find_homologs(logger, all_train_records, all_test_records, output_dir / 'train_test_homologs.m8',
                  output_dir / 'tmp', min_e_value, min_coverage, min_identity)


def main():
    # Init: Parse arguments, and create outputs directory and logger, and read sequences.
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir_name', default='data_processing', help='Directory name to save outputs')
    parser.add_argument('--min_e_value', default=1e-4, type=float, help='Minimum e-value for homologs/clustering')
    parser.add_argument('--min_coverage', default=0.6, type=float, help='Minimum coverage for homologs/clustering')
    parser.add_argument('--min_identity', default=0.5, type=float, help='Minimum identity for homologs/clustering')
    parser.add_argument('--test_ratio', default=0.2, type=float, help='Ratio of test data')
    args = parser.parse_args()

    outputs_dir = SCRIPT_DIR / args.outputs_dir_name
    outputs_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger(outputs_dir)

    logger.info('Reading sequences...')
    t4e_records = list(SeqIO.parse(T4E_PATH, 'fasta'))
    non_t4e_records = list(SeqIO.parse(NON_T4E_PATH, 'fasta'))
    logger.info(f'Done reading sequences. There are {len(t4e_records)} T4Es and {len(non_t4e_records)} non-T4Es sequences')

    # Step 1: Cut C-terminal.
    logger.info('Step 1: Cut C-terminal...')
    c_terminal_dir = outputs_dir / '1__C_terminal'
    c_terminal_dir.mkdir(exist_ok=True, parents=True)
    cut_C_terminal(logger, t4e_records, c_terminal_dir / 't4e.faa', 't4e')
    cut_C_terminal(logger, non_t4e_records, c_terminal_dir / 'non_t4e.faa', 'non_t4e')

    # Step 2: Create negative dataset.
    logger.info('Step 2: Create negative dataset...')
    negative_final_dir = outputs_dir / '2__Negative_Final'
    negative_final_dir.mkdir(exist_ok=True, parents=True)
    create_negative_final_dataset(logger, c_terminal_dir / 'non_t4e.faa', c_terminal_dir / 't4e.faa', negative_final_dir,
                                  args.min_e_value, args.min_coverage, args.min_identity)

    # Step 3: Cluster sequences and choose representatives.
    logger.info('Step 3: Cluster sequences and choose representatives...')
    representatives_dir = outputs_dir / '3__Representatives'
    representatives_dir.mkdir(exist_ok=True, parents=True)
    cluster_records(logger, c_terminal_dir / 't4e.faa', representatives_dir / 't4e',
                    representatives_dir / 't4e_representatives.faa',
                    args.min_e_value, args.min_coverage, args.min_identity)
    cluster_records(logger, negative_final_dir / 'negative_final.faa', representatives_dir / 'negative',
                    representatives_dir / 'negative_representatives.faa',
                    args.min_e_value, args.min_coverage, args.min_identity)

    # Step 4: Split to train and test.
    logger.info('Step 4: Split to train and test...')
    final_datasets_dir = outputs_dir / '4__Final_Datasets'
    final_datasets_dir.mkdir(exist_ok=True, parents=True)
    split_fasta(logger, representatives_dir / 't4e_representatives.faa', final_datasets_dir / 'positive_train_data.fasta',
                final_datasets_dir / 'positive_test_data.fasta', args.test_ratio)
    split_fasta(logger, representatives_dir / 'negative_representatives.faa', final_datasets_dir / 'negative_train_data.fasta',
                final_datasets_dir / 'negative_test_data.fasta', args.test_ratio)

    # Step 5: Verify non-homology between train and test
    logger.info('Step 5: Verify non-homology between train and test...')
    verify_non_homologs_dir = outputs_dir / '5__Verify_Train_Test_Non_Homologs'
    verify_non_homologs_dir.mkdir(exist_ok=True, parents=True)
    verify_train_and_test_non_homologs(logger, final_datasets_dir / 'positive_train_data.fasta',
                                       final_datasets_dir / 'positive_test_data.fasta',
                                       final_datasets_dir / 'negative_train_data.fasta',
                                       final_datasets_dir / 'negative_test_data.fasta', verify_non_homologs_dir,
                                       args.min_e_value, args.min_coverage, args.min_identity)

if __name__ == '__main__':
    main()
