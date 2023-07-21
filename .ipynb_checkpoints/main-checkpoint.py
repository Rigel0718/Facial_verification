from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from utils.Embedding import Embedding_vector, Embeddings_Manager, img_loader, evaluate_model
from utils.Label_DataFrame import Label_DataFrame, distance
import cv2
import numpy as np
import torch
from torchvision import transforms
import os


transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        fixed_image_standardization
    ])

img_path = '/opt/ml/data/example/group_less.jpg'


# detection model 생성
# keep_all을 제외하면 default
detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709, keep_all=True) 
single_detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709) 

# print(x.shape) # 5명인 경우 Tensor [5, 3, 160, 160], single은 하나만 추출 [3, 160, 160]

# # 모델 정의 
facenet_model = InceptionResnetV1(classify=False, pretrained='vggface2')

embedding_facenet = Embedding_vector(model=facenet_model)

enrolled_image = dict()  # 로그인 사람 마다 불러옴
label = 'people'  # 이미지 등록할 때 받아오는 키

# 이름과 이미지가 들어왔다.

def get_detected_images(image_path, detection_model : MTCNN) :
    img = img_loader(image_path)
    feature = detection_model(img)
    return feature

def get_image(image_path, label, store ,model : Embedding_vector, single_detection_model : MTCNN) :
    feature = get_detected_images(image_path, single_detection_model) # feature = [3, 160, 160]
    print(type(feature))
    embedding_vector = model.get_embedding(feature=feature)
    store[label] = embedding_vector
    # print(embedding_vector)
    return embedding_vector # numpy.array (1, 512)

def enrolling_image(label, store : dict, single_embedding_vector) :
    store[label] = single_embedding_vector
    # print(single_detection_model)
    # return single_embedding_vector
facenet_model.eval()
with torch.no_grad():
    embedding = get_image(img_path, label, store= enrolled_image, model=embedding_facenet, single_detection_model=single_detection_model)
# enrolled_image(label, store=enrolled_image, single_detection_model=embedding)
# print(enrolled_image)

# 원하는 사람 픽 
key = 'people'


def make_album(file_path, model : Embedding_vector, detection_model : MTCNN) :
    album_dict = dict()
    for dirname, _, filenames in os.walk(file_path) :
        for filename in filenames :
            image_path = os.path.join(dirname, filename)
            feature = get_detected_images(image_path, detection_model=detection_model)
            album_dict[image_path] = model.get_embedding(feature)
    return album_dict

def get_aa (key, threshold, album : dict) :
    key_image_set = set()
    key_image_embedding = enrolled_image[key]
    for image in album.keys() :
        embeddings = album[image] # (n, 512)
        for embedding in range(embeddings.shape[0]) :
            dis = distance(key_image_embedding, embedding, 1)
            if dis <= threshold : 
                key_image_set.add(image)
                break     # 한 명 찾았으니까 더 찾지 않고 break


# TODO enrolled image를 embedding vector와 저장하기
# TODO file image embedding vector와 저장하기
# enrolled image와 file image distance 구해서 threshold에 해당하는 이미지 따로 보관하기
# 그 이미지 묶어서 배출하기 