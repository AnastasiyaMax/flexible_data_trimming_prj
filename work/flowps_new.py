import argparse
import csv
import logging
import os
import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.kernel_ridge import KernelRidge

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
  format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
  level=logging.DEBUG
)
logger = logging.getLogger('flowps')

classifierList = {
  "KernelRidge": (KernelRidge, {}),
  "AdaBoost": (AdaBoostClassifier, {}),
  "BernoulliNB": (BernoulliNB, {"alpha":1.0, "binarize":0.0, "fit_prior":False}),
  "MLP": (MLPClassifier, {"hidden_layer_sizes":(30, ), "alpha":0.001}),
  "RandomForest": (RandomForestClassifier, {"n_estimators":30, "criterion":'entropy', "class_weight":'balanced_subsample'}),
  "LinearSVC": (svm.LinearSVC, {"C":1}),
  "SVC": (svm.SVC, {"kernel":'poly', "C":1}),
  "knn": (lambda x: "knn", {}),
}

class FlowPS(object):

  def __init__(self, min_neighbours, max_neighbours, min_surround, max_surround, clf):
    self.min_neighbours = min_neighbours
    self.max_neighbours = max_neighbours
    self.min_surround = min_surround
    self.max_surround = max_surround
    self.train = None
    # new
    self.features = None
    # new
    self.predictions = None
    self.clf = classifierList[clf][0](**classifierList[clf][1])

  def read_csv(self, train_csv):
    self.train = pd.read_csv(train_csv)
    self.predictions = np.zeros([
      self.max_surround - self.min_surround + 1,
      self.max_neighbours - self.min_neighbours + 1,
      self.train.shape[0],
    ])

  # new
  def read_features(self, f):
    with open(f, 'rb') as file:
      self.features = pickle.load(file)
  # new

  def _get_scaled_data_by_params(self, train_data, train_score, validation_data, surround, neighbours):
    """
    Returns normalized data as np.array
    """
    high_count = np.sum(validation_data < train_data, axis=0)
    low_count = np.sum(validation_data > train_data, axis=0)

    mask = ((high_count > surround) & (low_count > surround)).values

    validation_masked = validation_data[mask]
    train_masked = train_data.iloc[:, mask]

    diff = np.array(train_masked - validation_masked).astype(float)
    distances = np.linalg.norm(diff, axis=1)

    idx = np.argpartition(distances, neighbours - 1)

    neighbour_train_data = train_masked.iloc[idx[:neighbours], :].astype(np.float64)
    neighbour_train_score = train_score.iloc[idx[:neighbours]].astype(np.float64).values
    validation_test_data = validation_masked.values.reshape(1, -1).astype(np.float64)

    scaler = StandardScaler()
    neighbour_train_data = scaler.fit_transform(neighbour_train_data)
    validation_test_data = scaler.transform(validation_test_data)

    return neighbour_train_data, neighbour_train_score, validation_test_data

  def _calculate_prediction(self, surround, neighbours):

    prediction = np.zeros(self.train.shape[0])
    for vindex in range(self.train.shape[0]):
      # new
      validation_data = self.train.iloc[vindex, self.features[vindex]]
      train_data = self.train.drop(vindex).iloc[:, self.features[vindex]]
      # new
      train_score = self.train.drop(vindex).iloc[:, 1]
      train_x, train_y, test_x = self._get_scaled_data_by_params(
        train_data,
        train_score,
        validation_data,
        surround,
        neighbours
      )
      if(self.clf == "knn"):
        prediction[vindex] = np.mean(train_y)
      else:
        self.clf.fit(train_x, train_y)
        prediction[vindex] = self.clf.predict(test_x)[0]
    return prediction

  def calculate_predictions(self, out_file):
    writer = csv.writer(out_file)

    logger.debug('Calculate predictions')
    for i, surround in enumerate(range(self.min_surround, self.max_surround + 1)):
      logger.debug('Number of surrounding points: {}'.format(surround))
      for j, neighbours in enumerate(range(self.min_neighbours, self.max_neighbours + 1)):
        logger.debug('Number of neighbours: {}'.format(neighbours))
        self.predictions[i, j] = self._calculate_prediction(surround, neighbours)
        #logger.debug(format(self.predictions[i, j]))
        writer.writerow(np.round(self.predictions[i, j], decimals=4))

  def plot_auc(self, out_dir):
    train_score = self.train.iloc[:, 1].values > 50
    scores = np.zeros(self.predictions.shape[0:2])
    for i in range(scores.shape[0]):
      for j in range(scores.shape[1]):
        scores[i, j] = roc_auc_score(train_score, self.predictions[i, j])

    fig, ax = plt.subplots(figsize=(15, 3))
    cax = ax.imshow(
      np.flip(scores, axis=0),
      cmap=plt.cm.jet,
      extent=[self.min_neighbours, self.max_neighbours + 1, self.min_surround, self.max_surround + 1],
    )
    fig.colorbar(cax)
    ax.set_xlabel('K')
    ax.set_ylabel('M')
    ax.set_title('AUC as a function of number of surrounding points (M) and number of neighbours (K)')
    plt.savefig(os.path.join(out_dir, 'auc.png'))


def parse_arguments():
  parser = argparse.ArgumentParser(description='FlowPS')
  parser.add_argument('--min-surround', type=int, default=1, help='Minimum number of surrounding points')
  parser.add_argument('--max-surround', type=int, default=25, help='Maximum number of surrounding points')
  parser.add_argument('--min-neighbours', type=int, default=20, help='Minimum neighbours')
  parser.add_argument('--max-neighbours', type=int, default=226, help='Maximum neighbours')
  parser.add_argument('--train-file', type=str, required=True, help='CSV file path with train data')
  # new
  parser.add_argument('--features', type=str, required=True, help='pickle file path with features')
  # new
  parser.add_argument('--out-dir', type=str, help='Output directory')
  parser.add_argument('--clf', type=str, default='LinearSVC', help='Classifier type')
  parser.add_argument('--clf-args', type=str, default='{}', help='Classifier arguments')
  args = parser.parse_args()
  return args


def init_logger(out_dir):
  log_handler = logging.FileHandler(os.path.join(out_dir, 'log.txt'), mode='w')
  log_handler.setFormatter(logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s'))
  log_handler.setLevel(logging.DEBUG)
  logger.addHandler(log_handler)


def main():
  args = parse_arguments()

  jsonObject = json.loads(args.clf_args)
  for k in jsonObject:
    classifierList[args.clf][1][k] = jsonObject[k]

  if not args.out_dir:
    train_name = os.path.basename(args.train_file).split('.')[0]
    args.out_dir = os.path.join(PROJECT_PATH, 'results', train_name)

  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

  init_logger(args.out_dir)

  flowps = FlowPS(args.min_neighbours, args.max_neighbours, args.min_surround, args.max_surround, args.clf)
  flowps.read_csv(args.train_file)
  # new
  flowps.read_features(args.features)
  # new

  out_file = os.path.join(args.out_dir, 'predictions.csv')
  with open(out_file, 'w') as f:
    flowps.calculate_predictions(out_file=f)

  flowps.plot_auc(args.out_dir)


if __name__ == '__main__':
  main()
