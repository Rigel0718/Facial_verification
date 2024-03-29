import torch
import cv2 
import os
import numpy as np
import math
# from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from collections import defaultdict

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


def img_loader(path) :
    try : 
        with open(path, 'rb') as f :
            img = cv2.imread(path) 
            if len(img.shape) == 2 :
                img = np.stack([img] * 3, 2)
        
            return img
    except IOError :
        print('Cannot load image' + path)


def get_embedding(model, image_path=None, feature=None) :
        
        # image_path로 기입된 경우와 image자체로 기입된 경우를 나눈다.
        if image_path is not None :
            img = img_loader(image_path)
        else : 
            img = feature

        # single 이미지는 3차원이기 때문에 모델에 넣어줄 수 있게 4차원 변환
        if len(img.shape) == 3:         
            img = img.unsqueeze(0)
            
        embedding = model(img)
        embedding = embedding.to('cpu').numpy()   # embedding 연산은 CPU로 진행하기 위해 변환

        return embedding

# 일반 사진에서 얼굴을 찾아서 return 
def get_detected_images(image_path, detection_model : MTCNN) :
    img = img_loader(image_path)
    feature = detection_model(img)
    return feature    # feature :  5명인 경우 Tensor [5, 3, 160, 160], single은 하나만 추출 [3, 160, 160]

def get_embedding_vector_store_dict(store : dict, label, features : torch.Tensor, model ) :
    embedding_vector = get_embedding(model, feature=features)
    store[label] = embedding_vector

# album이 폴더 형태로 정리 되어있을 때
# 그 폴더를 album으로 새로 만든다.
def make_album (folder_path : list, model, detection_model : MTCNN) :
    album_dict = dict()
    for image_path in folder_path :
        feature = get_detected_images(image_path, detection_model=detection_model)
        if feature is None :
            print(image_path)
            print('Cannot Detect Face on the image')
            continue
        get_embedding_vector_store_dict(album_dict, image_path, feature, model)
    return album_dict   # {image_1.jpg : (n,512),}


def make_album_folder(folder_path, model, detection_model : MTCNN) :
    album_dict = dict()
    for dirname, _, filenames in os.walk(folder_path) :
        for filename in filenames :
            image_path = os.path.join(dirname, filename)
            feature = get_detected_images(image_path, detection_model=detection_model)
            if feature is None :
                # print(image_path)
                continue
            get_embedding_vector_store_dict(album_dict, image_path, feature, model)
    return album_dict   # {image_1.jpg : (n,512),}

# 거리를 비교해서, 적절한 image를 set형태로 return
def get_result (key, enrolled_DB, threshold, album : dict) :
    key_image_set = set()
    key_image_embedding = enrolled_DB[key]
    for image in album.keys() :
        embeddings = album[image] # (n, 512)
        dis = distance(key_image_embedding, embeddings, 1)
        # print(image, dis)
        result = dis <= threshold  # result [True, False, False] 
        if any(result) :          # any => 하나라도 True가 있으면
            key_image_set.add(image)

    return key_image_set

def get_score(result : set) :
    score = defaultdict(lambda : 0)
    for image_path in result :
        if 'iu' in image_path :
            score['correct'] += 1
        else :
            score['incorrect'] += 1
    return score

# detection model 생성
# keep_all을 제외하면 default
detection_model = MTCNN(image_size=160, margin=0.6, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709, keep_all=True) 
single_detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709) 

facenet_model = InceptionResnetV1(classify=False, pretrained='vggface2')
##########################
# model_path = '/opt/ml/insightface/recognition/arcface_torch/work_dirs/final/sgd 0.0001_160_f_best.pt'
model_path = '/opt/ml/Facial_verification/workspace/re_bat64_lr_7e-4_20230815_154634/final.pt'
facenet_model.load_state_dict(torch.load(model_path))


# threshold = 0.3046  # cosine threshold
# threshold = 0.847   # euclidian threshold
# threshold = 0.37769  # pth
# threshold = 1.250199

# model_path = '/opt/ml/Facial_verification/workspace/re_bat64_lr_7e-4_20230815_154634/final.pt'
# threshold = 1.2514000134468082 # euclidian threshold
threshold = 0.37819998884201056 # cosine


# 앨범 이미지 추가
facenet_model.eval()
with torch.no_grad():
    # file_path = ['/opt/ml/data/example/album/2_ol.jpg', '/opt/ml/data/example/album/dog_1.jpg', '/opt/ml/data/example/album/group_less.jpg',
    #             '/opt/ml/data/example/album/iu_1.jpg', '/opt/ml/data/example/album/iu_kara.jpg', '/opt/ml/data/example/album/hyojin.jpeg',
    #             '/opt/ml/data/example/album/boyoung_2.jpg','/opt/ml/data/example/album/han_1.jpg','/opt/ml/data/example/album/iu_2.jpg',
    #             '/opt/ml/data/example/album/huh.jpg', '/opt/ml/data/example/album/kazha.jpg','/opt/ml/data/example/album/huh_2.jpg']
    # album = make_album(file_path, facenet_model, detection_model)
    file_path = '/opt/ml/data/example/album'
    album = make_album_folder(file_path, facenet_model, detection_model)


# 찾고싶은 이미지 입력
facenet_model.eval()
with torch.no_grad():
    image_path = ['/opt/ml/data/example/enrolled/iu_base.jpg']
    # image_path = ['/opt/ml/data/example/enrolled/one_bin.jpeg']
    
    # 찾을 사람을 저장하는 dictionary 
    key_dict = make_album(image_path, facenet_model, single_detection_model)


# 결과 찾기
DB = album
key_image = '/opt/ml/data/example/enrolled/iu_base.jpg'
# key_image = '/opt/ml/data/example/enrolled/one_bin.jpeg'
final_classification = get_result(key_image, key_dict, threshold, DB)
score = get_score(final_classification)

print('final : ', final_classification)
print('score : ', score)