from Bio import SeqIO
from sklearn.model_selection import train_test_split

from .consts import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                    FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                    RANDOM_STATE)


def read_fasta_file(file_path):
    sequences = {}
    for seq_record in SeqIO.parse(file_path, 'fasta'):
        sequences[seq_record.id] = str(seq_record.seq)
    return sequences


def read_sequences_from_fasta_file(file_path):
    id_to_sequence = read_fasta_file(file_path)
    return list(id_to_sequence.values())


def read_train_without_split():
    positive_train = read_sequences_from_fasta_file(FIXED_POSITIVE_TRAIN_FILE)
    negative_train = read_sequences_from_fasta_file(FIXED_NEGATIVE_TRAIN_FILE)

    all_train_labels = [1] * len(positive_train) + [0] * len(negative_train)
    all_train_sequences = positive_train + negative_train

    return all_train_sequences, all_train_labels


def read_train_data():
    all_train_sequences, all_train_labels = read_train_without_split()

    train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(
        all_train_sequences, all_train_labels, test_size=0.25, random_state=RANDOM_STATE, shuffle=True, stratify=all_train_labels)

    return train_sequences, validation_sequences, train_labels, validation_labels


def read_test_data(split):
    if split == 'test':
        positive_test = read_sequences_from_fasta_file(FIXED_POSITIVE_TEST_FILE)
        negative_test = read_sequences_from_fasta_file(FIXED_NEGATIVE_TEST_FILE)
    else:
        raise ValueError(f"split must be one of ['test'], got {split}")

    test_labels = [1] * len(positive_test) + [0] * len(negative_test)
    test_sequences = positive_test + negative_test

    return test_sequences, test_labels