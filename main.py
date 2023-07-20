from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from utils.Embedding import Embedding_vector, Embeddings_Manager
from utils.Label_DataFrame import Label_DataFrame
import cv2
import numpy as np

def img_loader(path) :
    try : 
        with open(path, 'rb') as f :
            img = cv2.imread(path) 
            if len(img.shape) == 2 :
                img = np.stack([img] * 3, 2)
        
            return img
    except IOError :
        print('Cannot load image' + path)

detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709, keep_all=True)
img_path = '/opt/ml/data/example/group_less.jpg'
img = img_loader(img_path)
x = detection_model(img)
print(x.shape)

extraction_model = InceptionResnetV1(classify=False, pretrained='vggface2')
# enrolled_img_path = ''
