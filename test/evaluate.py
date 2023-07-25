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

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn

def calculate_roc(thresholds, embedding1, embedding2, actual_issame, distance_metric, ss) :
    assert (embedding1.shape[1]==embedding2.shape[1])  #  embedding vector size 
    nrof_pairs = embedding2.size(0)
    nrof_thresholds = len(thresholds)

    acc_train = np.zeros((nrof_thresholds))
    for threshold in nrof_thresholds :   ############
        adsf = threshold

        
    tprs = np.zeros((nrof_thresholds))  # true positive rate
    fprs = np.zeros((nrof_thresholds))  # false positive rate
    accuracy = 0   # 일단 만들어놈

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)  # batch에 들어가는 정답 pair들의 index설정

def evaluate(embeddings, enrolled_embedding, label_bool ,thresholds, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    embedding = enrolled_embedding
    embeddings2 = embeddings
    actual_issame = [label_bool] * embeddings.size(0)
    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embedding, embeddings2,
        np.asarray(actual_issame), distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    return tpr, fpr, accuracy, fp, fn
     


