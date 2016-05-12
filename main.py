import numpy as np

settings = {"labels": "D:\\Datasets\\retinopathy\\trainLabels.csv", "labels": "D:\\Datasets\\retinopathy\\trainLabels.csv"}

def load_labels(path):
    np.loadtxt(path, str, skiprows=1, delimiter=",").tolist()
