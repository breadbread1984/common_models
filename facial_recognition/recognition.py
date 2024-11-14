#!/usr/bin/python3

from os.path import exists, join
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor
from torchvision.datasets import CelebA
from facenet_pytorch import MTCNN, InceptionResnetV1
from db import QuantizedDB

class Recognition(object):
  def __init__(self, device = 'cuda'):
    self.db = QuantizedDB.create(hidden_dim = 512)
    self.mtcnn = MTCNN(image_size = 160, margin = 0, min_face_size = 20,
                       thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                       device = device).eval()
    self.resnet = InceptionResnetV1(classify = False, pretrained = 'vggface2').to(device).eval()
    self.device = device
    self.labels = None
  def test_celeba(self, batch_size = 1024):
    # 1) vectorize images and save into database
    print('building database of known faces...')
    if not exists('index_file.ivfpq'):
      trainset = CelebA(root = 'celeba', split = 'train', target_type = 'identity', download = True)
      batch = list()
      labels = list()
      for img, label in tqdm(trainset):
        # NOTE: img is PIL image
        x_aligned, prob = self.mtcnn(img, return_prob = True)
        # x_aligned range in [-1,1], shape = (3, 160, 160) in RGB order
        if x_aligned is None: continue
        batch.append(x_aligned.detach())
        labels.append(label.detach().cpu().numpy())
        if len(batch) == batch_size:
          aligned = torch.stack(batch).detach().to(self.device)
          embeddings = self.resnet(aligned)
          self.db.add(embeddings.detach().cpu().numpy())
          batch = list()
      if len(batch):
        aligned = torch.stack(batch).detach().to(self.device)
        embeddings = self.resnet(aligned)
        self.db.add(embeddings.detach().cpu().numpy())
      self.labels = np.array(labels) # label.shape = (sample_num,)
      self.save()
    # 2) match with K-nn
    print('recognition of unknown faces...')
    if exists('index_file.ivfpq'):
      self.load()
    evalset = CelebA(root = 'celeba', split = 'valid', target_type = 'identity', download = True)
    correct = 0
    total = 0
    batch = list()
    labels = list()
    for img, label in tqdm(evalset):
      x_aligned, prob = self.mtcnn(img, return_prob = True)
      if x_aligned is None: continue
      batch.append(x_aligned.detach())
      labels.append(label.detach().cpu().numpy())
      if len(batch) == batch_size:
        aligned = torch.stack(batch).detach().to(self.device)
        embeddings = self.resnet(aligned)
        D, I = self.db.match(embeddings.detach().cpu().numpy(), k = 5) # I.shape = (batch_size, 5)
        neighbors = self.labels[I] # true_labels.shape = (batch_size, 5)
        preds = list()
        for neighbor in neighbors:
          values, counts = np.unique(neighbor, return_counts = True)
          pred = values[np.argmax(counts)]
          preds.append(pred)
        correct += np.sum((np.array(labels) == np.array(preds)).astype(np.int32))
        total += batch_size
        batch = list()
        labels = list()
    if len(batch):
      aligned = torch.stack(batch).to(self.device)
      embeddings = self.resnet(aligned)
      D, I = self.db.match(embeddings.detach().cpu().numpy(), k = 5)
      neighbors = self.labels[I]
      preds = list()
      for neighbor in neighbors:
        values, counts = np.unique(neighbor, return_counts = True)
        pred = values[np.argmax(counts)]
        preds.append(pred)
      correct += np.sum((np.array(labels) == np.array(preds)).astype(np.int32))
      total += len(batch)
    print(f'accuracy: {correct / total}')
  def save(self, ):
    np.save('labels.npy', self.labels)
    self.db.serialize()
  def load(self, ):
    self.labels = np.load('labels.npy')
    self.db = QuantizedDB.deserialize()

if __name__ == "__main__":
  recog = Recognition()
  recog.test_celeba()
