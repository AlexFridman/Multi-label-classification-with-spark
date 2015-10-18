from pyspark import RDD

def shuffle_and_split(data: RDD, fold_n: int, seed = 0):
    fold_weights = [1 / fold_n] * fold_n
    return data.randomSplit(fold_weights)

def hold_out(sc, data: RDD, k: int, model_builder, metrics: list):
    folds = shuffle_and_split(data, k)
    for i in range(k):
        test = folds[i]
        training = sc.union(folds[:i] + folds[i + 1:])
        model = model_builder(training)
        lables_and_predictions = test.map(lambda x: (x['lables'], model.predict(x['features'])))
        for metric in metrics:
            metric.evaluate(lables_and_predictions)
    return metrics