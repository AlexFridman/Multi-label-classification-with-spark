from pyspark.mllib.classification import NaiveBayesModel
from pyspark.mllib.linalg import _convert_to_vector
from pyspark.mllib.linalg import Vectors
from pyspark import RDD
import numpy as np
import math

def scale(x: np.ndarray):
    mean_x = x.mean()
    max_x = x.max()
    min_x = x.min()
    return (x - min_x) / (max_x - min_x)

class MLNaiveBayesModel(NaiveBayesModel):
    def predict_all(self, x):
        if isinstance(x, RDD):
            return x.map(lambda v: self.predict_all(v))
        x = _convert_to_vector(x)
        log_probs = self.pi + x.dot(self.theta.transpose())
        scaled_log_probs = scale(log_probs)
        int_lables = [int(l_i) for l_i in self.labels]
        labels_and_log_probs = zip(int_lables, scaled_log_probs)
        return sorted(labels_and_log_probs, key=lambda x: x[1], reverse=True)
    
# RDD (labels) (features)
def train_model(data, l = 1.0):
    aggreagated = data.flatMap(lambda x: [(l, x['features']) for l in x['lables']]). \
        combineByKey(lambda v: (1, v),
                 lambda c, v: (c[0] + 1, c[1] + v),
                 lambda c1, c2: (c1[0] + c2[0], c1[1] + c2[1])). \
        sortBy(lambda x: x[0]). \
        collect()
    num_labels = len(aggreagated)
    num_documents = data.count()
    num_features = aggreagated[0][1][1].array.size
    labels = np.zeros(num_labels)
    pi = np.zeros(num_labels, dtype=int)
    theta = np.zeros((num_labels, num_features))
    pi_log_denom = math.log(num_documents + num_labels * l)
    i = 0
    for (label, (n, sum_term_freq)) in aggreagated:
        labels[i] = label
        pi[i] = math.log(n + l) - pi_log_denom
        theta_log_denom = math.log(sum_term_freq.toArray().sum() + num_features * l)
        for j in range(num_features):
            theta[i,j] = math.log(sum_term_freq[j] + l) - theta_log_denom
        i += 1  
    return MLNaiveBayesModel(labels, pi, theta)