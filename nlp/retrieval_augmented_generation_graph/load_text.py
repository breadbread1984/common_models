#!/usr/bin/python3

from os import walk
from os.path import join, exists, splitext
from tqdm import tqdm
from absl import flags, app
import numpy as np
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from models import TGI

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory')
  flags.DEFINE_string('tgi_host', default = 'http://localhost:8080/generate', help = 'host of TGI')
  flags.DEFINE_string('neo4j_host', default = 'bolt://localhost:7687', help = 'host')
  flags.DEFINE_string('neo4j_user', default = 'neo4j', help = 'user name')
  flags.DEFINE_string('neo4j_password', default = 'neo4j', help = 'password')
  flags.DEFINE_string('neo4j_db', default = 'neo4j', help = 'database')

def main(unused_argv):
  embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  vectordb = Neo4jVector(
    embedding = embedding,
    url = FLAGS.neo4j_host,
    username = FLAGS.neo4j_user,
    password = FLAGS.neo4j_password,
    database = FLAGS.neo4j_db,
    index_name = "typical_rag",
    search_type = "hybrid",
    pre_delete_collection = True
  )
  # index creating code is only available in Neo4jVector.from_documents
  # but it should be called manually when you want to incrementally add documents
  embedding_dimension, index_type = vectordb.retrieve_existing_index()
  if not index_type:
    vectordb.create_new_index()
  # load
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 150, chunk_overlap = 10)
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext.lower() in ['.htm', '.html']:
        loader = UnstructuredHTMLLoader(join(root, f))
      elif ext.lower() == '.txt':
        loader = TextLoader(join(root, f))
      elif ext.lower() == '.pdf':
        loader = UnstructuredPDFLoader(join(root, f), mode = 'single')
      else:
        raise Exception('unknown format!')
      docs = loader.load()
      split_docs = text_splitter.split_documents(docs)
      for doc in split_docs:
        doc.metadata['url'] = f'file://{join(root,f)}'
        doc.metadata['classification'] = np.random.randint(low = 0, high = 3)
      vectordb.add_documents(split_docs)

if __name__ == "__main__":
  add_options()
  app.run(main)
