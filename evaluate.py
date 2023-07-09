import numpy as np
import math

def distance(embedding1, embedding2, distance_metric='euclidian') :
    if distance_metric.lower() == 'euclidian' :
        diff = np.subtract(embedding1, embedding2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric.lower() == 'cosine' :
        dot = np.sum(np.multiply(embedding1,embedding2), axis=1)
        norm = np.linalg.norm(embedding1,axis=1) * np.linalg.norm(embedding2,axis=1)
        similarity = dot/norm
        dist = np.arccos(similarity) / math.pi
    else :
        raise 'Undefined distance metric please enter euclidian or cosine'
    
    return dist

def calculate_roc(thresholds, embedding1, embedding2, actual_issame, distance_metric, ss) :
    assert (embedding1.shape[0]==embedding2.shape[0])
    assert (embedding1.shape[1]==embedding2.shape[1])
    nrof_pairs = min(len(actual_issame), embedding1.shape[0])
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof))