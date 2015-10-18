from pyspark import RDD

class Metric:
    def __init__(self, name: str, verbose=False):
        self._name = name
        self._results = []
        self._verbose = verbose
        
    @property
    def name(self):
        return self._name
    
    @property
    def results(self):
        return self._results
    
    @property
    def avg(self):
        return np.average(_results)
    
    def evaluate(self, lables, predictions):
        pass

class AccuracyMetric(Metric):
    def __init__(self, pred_n: int, intersect_n: int):
        self._pred_n = pred_n
        self._intersect_n = intersect_n
        super(AccuracyMetric, self).__init__(name='Accuracy', verbose=False)
        
    def evaluate(self, lables_and_predictions: RDD):
        TP = lables_and_predictions.map(lambda x:
                                    (set(x[0]), set([p for p,w in x[1][:self._pred_n]]))). \
                                    filter(lambda x:
                                           len(x[0].intersection(x[1])) > self._intersect_n)
        accuracy = 100.0 * TP.count() / lables_and_predictions.count()
        if self._verbose:
            print('accuracy: ', accuracy)
        self._results.append(accuracy)
        return accuracy

class HammingMetric(Metric):
    def __init__(self, pred_n: int, intersect_n: int):
        self._pred_n = pred_n
        self._intersect_n = intersect_n
        super(AccuracyMetric, self).__init__(name='Hamming', verbose=False)
        
    def _hamming_loss(true: list, pred: list):
        set_true = set(true)
        set_pred = set(pred)
        if len(set_true) == 0 and len(set_pred) == 0:
            return 1
        else:
            return len(set_true.intersection(set_pred)) / \
                len(set_true.union(set_pred))
        
    def evaluate(self, lables_and_predictions: RDD):
        result = lables_and_predictions.map(lambda p: _hamming_loss(p[0], p[1])). \
            mean()
        self._results.append(result)
        return result