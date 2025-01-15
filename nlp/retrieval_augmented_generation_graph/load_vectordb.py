#!/usr/bin/python3

from os import environ, walk
from os.path import join, exists, splitext
from tqdm import tqdm
from absl import flags, app
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from configs import neo4j_host, neo4j_user, neo4j_password, neo4j_db

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory')
  flags.DEFINE_integer('length', default = 150, help = 'segment length after splitting')
  flags.DEFINE_integer('overlap', default = 10, help = 'segment overlapping length')

def main(unused_argv):
  embedding = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-base")
  vectordb = Neo4jVector(
    embedding = embedding,
    url = neo4j_host,
    username = neo4j_user,
    password = neo4j_password,
    database = neo4j_db,
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
  text_splitter = RecursiveCharacterTextSplitter(separators = [r"\n\n", r"\n", r"\.(?![0-9])|(?<![0-9])\.", r"。"], is_separator_regex = True, chunk_size = FLAGS.length, chunk_overlap = FLAGS.overlap)
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
        doc.metadata['url'] = f'file://{join(root, f)}' # NOTE: psuedo document url
        import numpy as np
        doc.metadata['classification'] = np.random.randint(low = 0, high = 3) # NOTE: psuedo document classification
      split_docs = [doc for doc in split_docs if len(doc.page_content) > 3]
      vectordb.add_documents(split_docs)

if __name__ == "__main__":
  add_options()
  app.run(main)
