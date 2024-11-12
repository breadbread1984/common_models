#!/usr/bin/python3

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

if __name__ == "__main__":
  import numpy as np
  db = DB.create(256, dist = 'l2')
  samples = np.random.normal(size = (100,256))
  db.add(samples)
  samples = np.random.normal(size = (10,256))
  D, I = db.match(samples, k = 2)
  db.serialize()
  db2 = DB.deserialize()
