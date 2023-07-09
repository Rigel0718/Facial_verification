from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

input = torch.Tensor(1, 3, 250, 250)

resnet = InceptionResnetV1(pretrained='vggface2').eval()
ca_resnet = InceptionResnetV1(pretrained='casia-webface').eval()
embedding = resnet(input)
print(type(embedding))
print(embedding.shape)

embed = ca_resnet(input)
print(embed.shape)