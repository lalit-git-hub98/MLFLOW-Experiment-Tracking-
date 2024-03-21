import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging
import warnings


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average='weighted')
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    return acc, f1, precision, recall


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        data = pd.read_csv("glass.csv")
    except Exception as e:
        logger.exception(
            "Not able to read the csv file. Error: %s", e
        )

    
    train, test = train_test_split(data)

    
    train_x = train.drop(["Type"], axis=1)
    test_x = test.drop(["Type"], axis=1)
    train_y = train[["Type"]]
    test_y = test[["Type"]]

    HLS = 90
    maxIter = 3300

    with mlflow.start_run():
        mc = MLPClassifier(hidden_layer_sizes=HLS, max_iter=maxIter, random_state=42)
        mc.fit(train_x, train_y)

        predicted_qualities = mc.predict(test_x)

        (acc, f1, precision, recall) = eval_metrics(test_y, predicted_qualities)

        print("Multi-Layer Perceptron Classifier (HLS={:f}, maxIter={:f}):".format(HLS, maxIter))
        print("  ACC: %s" % acc)
        print("  f1: %s" % f1)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)

        mlflow.log_param("Hidden_Layer_Size", HLS)
        mlflow.log_param("Iterations", maxIter)
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)


        ## For Remote server only(DAGShub)

        # remote_server_uri="https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
        # mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                mc, "model", registered_model_name="MLPClassifierForGlass"
            )
        else:
            mlflow.sklearn.log_model(mc, "model")