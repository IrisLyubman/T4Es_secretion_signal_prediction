import os
import logging
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
import joblib
import json
import sys
from sklearn import __version__ as sklearn_version

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import FINAL_RESULTS
from src.pretrained_embeddings.calc_pt5_embeddings import calc_pt5_embeddings


def prepare_Xs_and_Ys(logger, model_id, split, always_calc_embeddings):
    Xs_positive, Xs_negative = calc_pt5_embeddings(model_id, split, always_calc_embeddings=always_calc_embeddings)

    Xs = np.concatenate([Xs_positive, Xs_negative])
    Ys = [1] * Xs_positive.shape[0] + [0] * Xs_negative.shape[0]

    # Shuffle
    combined = list(zip(Xs, Ys))
    random.shuffle(combined)
    shuffled_Xs, shuffled_Ys = zip(*combined)
    shuffled_Xs = np.array(shuffled_Xs)

    logger.info(f"Loaded {split} data: Xs_{split}.shape = {Xs.shape}, Ys_{split}.shape = {len(Ys)}")

    return shuffled_Xs, shuffled_Ys


def fit_on_data(Xs, Ys, output_dir):
    logging.info(f"Training Classifier with hyperparameters fixed")

    clf = MLPClassifier(activation='tanh', alpha=0.0001, early_stopping=True, hidden_layer_sizes=(50, 10),
                        learning_rate='constant', max_iter=400, random_state=500, solver='adam')
    clf.fit(Xs, Ys)

    # Save the best classifier to disk
    joblib.dump(clf, os.path.join(output_dir, "model.pkl"))
    # Save metadata
    metadata = {
        'numpy_version': np.__version__,
        'joblib_version': joblib.__version__,
        'sklearn_version': sklearn_version
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f)


def main():
    model_id = 'pt5'
    output_dir = os.path.join(FINAL_RESULTS, 'trained_pt5_head')
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            os.path.join(output_dir, 'classification_with_classic_ML.log'), mode='w')])
    logger = logging.getLogger(__name__)

    Xs_train, Ys_train = prepare_Xs_and_Ys(logger, model_id, 'train', always_calc_embeddings=False)
    Xs_test, Ys_test = prepare_Xs_and_Ys(logger, model_id, 'test', always_calc_embeddings=False)

    Xs = np.concatenate([Xs_train, Xs_test])
    Ys = np.concatenate([Ys_train, Ys_test])

    fit_on_data(Xs, Ys, output_dir)
    logging.info(f"Finished training classifier on embeddings for model {model_id}")


if __name__ == "__main__":
    main()
