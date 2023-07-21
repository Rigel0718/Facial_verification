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

img_path = '/opt/ml/data/example/enrolled/iu_base.jpg'
# img_path = '/opt/ml/data/example/enrolled/one_bin.jpeg'
file_path = '/opt/ml/data/example/album'
# threshold = 0.3046
threshold = 0.847

# detection model 생성
# keep_all을 제외하면 default
detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709, keep_all=True) 
single_detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709) 

# print(x.shape) # 5명인 경우 Tensor [5, 3, 160, 160], single은 하나만 추출 [3, 160, 160]

# # 모델 정의 

# model_path = '/opt/ml/insightface/recognition/arcface_torch/work_dirs/r50_epoch20_fine_tuning/20model.pt'
facenet_model = InceptionResnetV1(classify=False, pretrained='vggface2')
# facenet_model.load_state_dict(torch.load(model_path))
embedding_facenet = Embedding_vector(model=facenet_model)

# 이름과 이미지가 들어왔다.

def get_detected_images(image_path, detection_model : MTCNN) :
    img = img_loader(image_path)
    # print(img.shape)
    feature = detection_model(img)
    return feature

def get_image(store, label, features : torch.Tensor ,model : Embedding_vector, ) :
    embedding_vector = model.get_embedding(feature=features)
    store[label] = embedding_vector
    # print(embedding_vector)
    # return embedding_vector # numpy.array (1, 512)


# 원하는 사람 픽 
key = 'people'


def make_album(file_path, model : Embedding_vector, detection_model : MTCNN) :
    album_dict = dict()
    for dirname, _, filenames in os.walk(file_path) :
        for filename in filenames :
            image_path = os.path.join(dirname, filename)
            feature = get_detected_images(image_path, detection_model=detection_model)
            if feature is None :
                print(image_path)
                continue
            album_dict[image_path] = model.get_embedding(feature=feature) # {image_1.jpg : (n,512),}
    return album_dict

def get_result (key, threshold, album : dict) :
    key_image_set = set()
    key_image_embedding = enrolled_image[key]
    for image in album.keys() :
        embeddings = album[image] # (n, 512)
        # for embedding in range(embeddings.shape[0]) :
        dis = distance(key_image_embedding, embeddings, 0)
        print('bbbbb : ', dis.shape)
        print(image)
        print(dis, type(dis))
        result = dis <= threshold
        print('cccc : ', result)
        if any(result) :
            key_image_set.add(image)

    return key_image_set

facenet_model.eval()
with torch.no_grad():
    album = make_album(file_path, embedding_facenet, detection_model)
    features = get_detected_images(img_path, single_detection_model)

    label = 'IU'  # 이미지 등록할 때 받아오는 키
    enrolled_image = dict()  # 로그인 사람 마다 불러옴
    get_image(enrolled_image, label, features, model=embedding_facenet)
    final_classification = get_result(label, threshold, album)

print(album.keys())
print('final : ', final_classification)
