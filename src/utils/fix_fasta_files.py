from pathlib import Path
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from consts import (ORIGINAL_POSITIVE_TRAIN_FILE, ORIGINAL_NEGATIVE_TRAIN_FILE, ORIGINAL_POSITIVE_TEST_FILE,
                    ORIGINAL_NEGATIVE_TEST_FILE, ORIGINAL_POSITIVE_XANTOMONAS_FILE, ORIGINAL_NEGATIVE_XANTOMONAS_FILE,
                    DATASETS_FIXED_DIR)


def fix_all_fasta_files():
    fix_fasta_file(ORIGINAL_POSITIVE_TRAIN_FILE)
    fix_fasta_file(ORIGINAL_NEGATIVE_TRAIN_FILE)
    fix_fasta_file(ORIGINAL_POSITIVE_TEST_FILE)
    fix_fasta_file(ORIGINAL_NEGATIVE_TEST_FILE)
    fix_fasta_file(ORIGINAL_POSITIVE_XANTOMONAS_FILE)
    fix_fasta_file(ORIGINAL_NEGATIVE_XANTOMONAS_FILE)


def fix_fasta_file(file_path):
    with open(file_path, "r") as f:
        records = list(SeqIO.parse(f, "fasta"))

    fixed_records = [SeqRecord(seq=record.seq[:100], id=record.id.replace("/", " ").replace("|", "_"), description="")
                     for record in records if "*" not in record.seq]

    fixed_file_path = os.path.join(DATASETS_FIXED_DIR, Path(file_path).name)
    SeqIO.write(fixed_records, fixed_file_path, "fasta")


if __name__ == "__main__":
    fix_all_fasta_files()
