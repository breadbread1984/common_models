#!/usr/bin/python3

from facenet_pytorch import MTCNN, InceptionResnetV1

class Recognition(object):
  def __init__(self, device = 'cuda'):
    self.mtcnn = MTCNN(device = device)
    self.resnet = InceptionResnetV1(classify = False, pretrained = 'vggface2').to(device)
  def load_faces(self, ):

