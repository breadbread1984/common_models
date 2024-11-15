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
    if self.gpu_index.metric_type == faiss.METRIC_INNER_PRODUCT:
      faiss.normalize_L2(samples)
    self.gpu_index.add(samples)
  def match(self, samples, k = 1):
    # NOTE: samples.shape = (sample_num, hidden_dim)
    assert samples.shape[1] == self.gpu_index.d
    if self.gpu_index.metric_type == faiss.METRIC_INNER_PRODUCT:
      faiss.normalize_L2(samples)
    D, I = self.gpu_index.search(samples, k) # D.shape = (sample_num, k) I.shape = (sample_num, k)
    return D, I

class QuantizedDB(object):
  def __init__(self, index = None, trainset = None, device = 'gpu'):
    assert index is not None
    assert device in {'cpu', 'gpu'}
    self.index = index
    self.trainset = np.zeros((0,self.index.d)) if trainset is None else trainset
    self.device = device
  def serialize(self, db_path = 'index_file.ivfpq'):
    if self.device == 'gpu':
      index = faiss.index_gpu_to_cpu(self.index)
    index_bytes = faiss.serialize_index(index)
    with open(db_path, 'wb') as f:
      pickle.dump(index_bytes, f)
      pickle.dump(self.trainset, f)
      pickle.dump(self.device, f)
  @classmethod
  def deserialize(cls, db_path = 'index_file.ivfpq'):
    with open(db_path, 'rb') as f:
      index_bytes = pickle.load(f)
      trainset = pickle.load(f)
      device = pickle.load(f)
    index = faiss.deserialize_index(index_bytes)
    if device == 'gpu':
      res = faiss.StandardGpuResources()
      index = faiss.index_cpu_to_gpu(res, 0, index)
    return cls(index = index, trainset = trainset, device = device)
  @classmethod
  def create(cls, hidden_dim, dist = 'ip', nlist = 100, m = 8, nprobe = 10, device = 'gpu'):
    dists = {
      'ip': faiss.IndexFlatIP,
      'l2': faiss.IndexFlatL2,
    }
    quantizer = dists[dist](hidden_dim)
    index = faiss.IndexIVFPQ(quantizer, hidden_dim, nlist, m, 8)
    if device == 'gpu':
      res = faiss.StandardGpuResources()
      index = faiss.index_cpu_to_gpu(res, 0, index)
    index.nprobe = nprobe
    return cls(index = index, device = device)
  def add(self, samples):
    # NOTE: samples.shape = (sample_num, hidden_dim)
    assert samples.shape[1] == self.index.d
    if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
      faiss.normalize_L2(samples)
    if self.index.is_trained == False:
      # collect trainset if index is not trained
      self.trainset = np.concatenate([self.trainset, samples], axis = 0)
      if self.trainset.shape[0] >= 10000:
        # train quantizer if trainset is enough to train index
        self.index.train(self.trainset)
        self.index.add(self.trainset)
        self.trainset = None
    else:
      # direct add samples to trained index
      self.index.add(samples)
  def match(self, samples, k = 1):
    # NOTE: samples.shape = (sample_num, hidden_dim)
    assert samples.shape[1] == self.index.d
    assert self.index.is_trained, f"feed over 10000 samples to train the index before do matching"
    if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
      faiss.normalize_L2(samples)
    D, I = self.index.search(samples, k) # D.shape = (sample_num, k) I.shape = (sample_num, k)
    return D, I

if __name__ == "__main__":
  from os.path import exists, join
  from wget import download
  import tarfile
  import numpy as np
  if not exists('siftsmall.tar.gz'):
    download('ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz', out = 'siftsmall.tar.gz')
  file = tarfile.open('siftsmall.tar.gz')
  file.extractall()
  file.close()
  def read_fvecs(file):
    with open(file, 'rb') as f:
      while True:
        dim_bytes = f.read(4)
        if not dim_bytes:
          break
        dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
        vector = np.frombuffer(f.read(4 * dim), dtype=np.float32)
        yield vector
  def read_ivecs(file):
    with open(file, 'rb') as f:
      while True:
        dim_bytes = f.read(4)
        if not dim_bytes:
          break
        dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
        vector = np.frombuffer(f.read(4 * dim), dtype=np.int32)
        yield vector
  base = list(read_fvecs(join('siftsmall', 'siftsmall_base.fvecs')))
  query = list(read_fvecs(join('siftsmall', 'siftsmall_query.fvecs')))
  ground_truth = list(read_ivecs(join('siftsmall', 'siftsmall_groundtruth.ivecs')))
  base = np.array(base)
  query = np.array(query)
  ground_truth = np.array(ground_truth)
  def compute_accuracy(I, ground_truth, k = 5):
    correct = (I == ground_truth[:, :k]).sum()
    total = ground_truth.shape[0] * k
    return correct / total
  # index
  db = DB.create(128, dist = 'l2')
  db.add(base)
  D, I = db.match(query, k = 5)
  print('faiss.Index accuracy: ', compute_accuracy(I, ground_truth, k = 5))
  # quantized index
  db = QuantizedDB.create(128, dist = 'l2')
  db.add(base)
  D, I = db.match(query, k = 5)
  print('faiss.IndexIVFPQ accuracy: ', compute_accuracy(I, ground_truth, k = 5))
