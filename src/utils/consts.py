import os
from pathlib import Path

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS_DIR = os.path.join(PROJECT_BASE_DIR, 'new_data', 'data_processing', '4__Final_Datasets')
PRETRAINED_MODELS_DIR = os.path.join(PROJECT_BASE_DIR, 'models')

OUTPUTS_DIR = os.path.join(PROJECT_BASE_DIR, 'outputs_optimization')
EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, 'pretrained_embeddings')
CLASSIFIERS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, 'embeddings_classifiers')
FINETUNED_MODELS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, 'finetuned_models')
FINAL_RESULTS = os.path.join(PROJECT_BASE_DIR, 'final_results')
USE_LOCAL_MODELS = True

MODEL_ID_TO_MODEL_NAME = {
    #'esm_6': Path(PRETRAINED_MODELS_DIR) / 'esm2_t6_8M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t6_8M_UR50D',
    # 'esm_12': Path(PRETRAINED_MODELS_DIR) / 'esm2_t12_35M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t12_35M_UR50D',
    # 'esm_30': Path(PRETRAINED_MODELS_DIR) / 'esm2_t30_150M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t30_150M_UR50D',
    # 'esm_33': Path(PRETRAINED_MODELS_DIR) / 'esm2_t33_650M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t33_650M_UR50D',
    # 'esm_36': Path(PRETRAINED_MODELS_DIR) / 'esm2_t36_3B_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t36_3B_UR50D',
    # 'protein_bert': Path(PRETRAINED_MODELS_DIR) / 'protein_bert',
    'pt5': Path(PRETRAINED_MODELS_DIR) / 'prot_t5_xl_uniref50_01_08_2024' if USE_LOCAL_MODELS else 'Rostlab/prot_t5_xl_uniref50',
    # 'pt5_half_precision': Path(PRETRAINED_MODELS_DIR) / 'prot_t5_xl_half_uniref50_01_08_2024' if USE_LOCAL_MODELS else 'Rostlab/prot_t5_xl_half_uniref50-enc',
}

MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION = {
    'esm_6': 8,
    'esm_12': 35,
    'esm_30': 150,
    'esm_33': 650,
    'esm_36': 3000,
    'protein_bert': 16,
    'pt5': 3000,
    # 'pt5_half_precision': 3000,
}

BATCH_SIZE = 8
RANDOM_STATE = 42
FINETUNE_NUMBER_OF_EPOCHS = 10
PROTEIN_BERT_MODEL_NAME = 'epoch_92400_sample_23500000.pkl'


FIXED_POSITIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "positive_train_data.fasta")
FIXED_NEGATIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "negative_train_data.fasta")

FIXED_POSITIVE_TEST_FILE = os.path.join(DATASETS_DIR, "positive_test_data.fasta")
FIXED_NEGATIVE_TEST_FILE = os.path.join(DATASETS_DIR, "negative_test_data.fasta")

