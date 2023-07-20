from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from utils.Embedding import Embedding_vector, Embeddings_Manager, img_loader
from utils.Label_DataFrame import Label_DataFrame
import cv2
import numpy as np

img_path = '/opt/ml/data/example/group_less.jpg'

# detection model 생성
# keep_all을 제외하면 default
detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709, keep_all=True) 

img = img_loader(img_path)
x = detection_model(img)
print(x.shape) # 5명인 경우 Tensor [5, 3, 160, 160]

# 모델 정의 
facenet_model = InceptionResnetV1(classify=False, pretrained='vggface2')

embedding_facenet = Embedding_vector(model=facenet_model)
embdding_vectors = embedding_facenet.get_embedding(feature=x)
print(embdding_vectors.shape)

