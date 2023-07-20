import numpy as np
import math
import pandas as pd
import itertools
# from cheff import bring

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

class Label_DataFrame :
    def __init__(self, identities : dict) :
        self.identities = identities
    
    def get_positive_df(self) :
        positives = []
        for key, values in self.identities.items():
        # print(values)
            for i in range(0, len(values)-1):
                for j in range(i+1, len(values)):
                    positive = []
                    positive.append(values[i])
                    positive.append(values[j])
                    positives.append(positive)

        positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
        positives["decision"] = "Yes"
        self.positives = positives
        return positive
    
    def get_negative_df(self) :
        samples_list = list(self.identities.values())

        negatives = []
        for i in range(0, len(self.identities) - 1):
            for j in range(i+1, len(self.identities)):
                cross_product = itertools.product(samples_list[i], samples_list[j])
                cross_product = list(cross_product)

            for cross_sample in cross_product:
                negative = []
                negative.append(cross_sample[0])
                negative.append(cross_sample[1])
                negatives.append(negative)

        negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
        negatives["decision"] = "No"
        self.negatives = negatives
        return negatives
    
    def concate(self) :
        df = pd.concat([self.positives, self.negatives]).reset_index(drop = True)
        self.data_frame = df
        return df
        
    def get_inference_df(self, path2embedding) :
        distance_list = []
        for row in self.data_frame.itertuples() :
            embedding_1 = path2embedding[row[1]]
            embedding_2 = path2embedding[row[2]]
            dist = round(distance(embedding_1, embedding_2, 0)[0],4)
        # print(dist)
            distance_list.append(dist)
        
        self.data_frame['distance'] = distance_list
        return self.data_frame
    
    def get_prediction_df(self, threshold) :
        self.data_frame['prediction'] = 'No'
        idx = self.data_frame[self.data_frame.distance <= threshold].index
        self.data_frame.loc[idx, 'prediction'] = 'Yes'
        return self.data_frame