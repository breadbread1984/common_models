#!/usr/bin/python3

from shutil import rmtree
from os.path import exists
import pickle
import numpy as np
import faiss

class DB(object):
  def __init__(self, gpu_index = None):
    assert gpu_index is not None
    self.gpu_index = gpu_index
  def serialize(self, db_path = 'index_file.faiss'):
    cpu_index = faiss.index_gpu_to_cpu(self.gpu_index)
    faiss.write_index(cpu_index, db_path)
  @classmethod
  def deserialize(cls, db_path = 'index_file.faiss'):
    res = faiss.StandardGpuResources()
    cpu_index = faiss.read_index(db_path)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return cls(gpu_index = gpu_index)
  @classmethod
  def create(cls, hidden_dim, dist = 'ip'):
    dists = {
      'ip': faiss.GpuIndexFlatIP,
      'l2': faiss.GpuIndexFlatL2,
    }
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    gpu_index = dists[dist](res, hidden_dim, flat_config)
    return cls(gpu_index = gpu_index)
  def add(self, samples):
    # NOTE: samples.shape = (sample_num, hidden_dim)
    assert samples.shape[1] == self.gpu_index.d
    faiss.normalize_L2(samples)
    self.gpu_index.add(samples)
  def match(self, samples, k = 1):
    # NOTE: samples.shape = (sample_num, hidden_dim)
    assert samples.shape[1] == self.gpu_index.d
    faiss.normalize_L2(samples)
    D, I = self.gpu_index.search(samples, k) # D.shape = (sample_num, k) I.shape = (sample_num, k)
    return D, I

class QuantizedDB(object):
  def __init__(self, gpu_index = None, trainset = None, nlist = 100, m = 8):
    assert gpu_index is not None
    self.gpu_index = gpu_index
    self.trainset = np.zeros((0,self.gpu_index.d)) if trainset is None else trainset
    self.nlist = nlist
    self.m = m
  def serialize(self, db_path = 'index_file.ivfpq'):
    cpu_index = faiss.index_gpu_to_cpu(self.gpu_index)
    index_bytes = faiss.serialize_index(cpu_index)
    with open(db_path, 'wb') as f:
      pickle.dump(index_bytes, f)
      pickle.dump(self.trainset, f)
      pickle.dump(self.nlist, f)
      pickle.dump(self.m, f)
  @classmethod
  def deserialize(cls, db_path = 'index_file.ivfpq'):
    with open(db_path, 'rb') as f:
      index_bytes = pickle.load(f)
      trainset = pickle.load(f)
      nlist = pickle.load(f)
      m = pickle.load(f)
    cpu_index = faiss.deserialize_index(index_bytes)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return cls(gpu_index = gpu_index, trainset = trainset, nlist = nlist, m = m)
  @classmethod
  def create(cls, hidden_dim, dist = 'ip', nlist = 100, m = 8):
    dists = {
      'ip': faiss.IndexFlatIP,
      'l2': faiss.IndexFlatL2,
    }
    quantizer = dists[dist](hidden_dim)
    cpu_index = faiss.IndexIVFPQ(quantizer, hidden_dim, nlist, m, 8)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    return cls(gpu_index = gpu_index, nlist = nlist, m = m)
  def add(self, samples):
    # NOTE: samples.shape = (sample_num, hidden_dim)
    assert samples.shape[1] == self.gpu_index.d
    faiss.normalize_L2(samples)
    if self.gpu_index.is_trained == False:
      # collect trainset if index is not trained
      self.trainset = np.concatenate([self.trainset, samples], axis = 0)
      if self.trainset.shape[0] >= self.gpu_index.nlist * self.gpu_index.cp.min_points_per_centroid:
        # train quantizer if trainset is enough to train index
        self.gpu_index.train(self.trainset)
        self.gpu_index.add(self.trainset)
        self.trainset = None
    else:
      # direct add samples to trained index
      self.gpu_index.add(samples)
  def match(self, samples, k = 1):
    # NOTE: samples.shape = (sample_num, hidden_dim)
    assert samples.shape[1] == self.gpu_index.d
    assert self.gpu_index.is_trained, f"feed over {self.gpu_index.d * self.gpu_index.cp.min_points_per_centroid} samples to train the index before do matching"
    faiss.normalize_L2(samples)
    D, I = self.gpu_index.search(samples, k) # D.shape = (sample_num, k) I.shape = (sample_num, k)
    return D, I

if __name__ == "__main__":
  import numpy as np
  db = QuantizedDB.create(256, dist = 'l2')
  samples = np.random.normal(size = (3900,256)).astype(np.float32)
  db.add(samples)
  samples = np.random.normal(size = (10,256)).astype(np.float32)
  D, I = db.match(samples, k = 2)
  db.serialize()
  db2 = QuantizedDB.deserialize()