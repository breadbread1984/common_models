#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from models import TGI
from prompts import extract_triplets_template
import config

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory')
  flags.DEFINE_string('tgi_host', default = 'http://localhost:8080/generate', help = 'host of TGI')
  flags.DEFINE_string('neo4j_host', default = 'bolt://localhost:7687', help = 'host')
  flags.DEFINE_string('neo4j_user', default = 'neo4j', help = 'user name')
  flags.DEFINE_string('neo4j_password', default = 'neo4j', help = 'password')
  flags.DEFINE_string('neo4j_db', default = 'neo4j', help = 'database')

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct') 
  prompt = extract_triplets_template(tokenizer, config.node_types, config.rel_types)
  llm = TGI(FLAGS.tgi_host)
  graph_transformer = LLMGraphTransformer(
    llm = llm,
    prompt = prompt,
    allowed_nodes = config.node_types,
    allowed_relationship = config.rel_types,
  )
  neo4j = Neo4jGraph(url = FLAGS.neo4j_host, username = FLAGS.neo4j_user, password = FLAGS.neo4j_password, database = FLAGS.neo4j_db)
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
      graph = graph_transformers.convert_to_graph_documents(docs)
      neo4j.add_graph_documents(graph)

if __name__ == "__main__":
  add_options()
  app.run(main)
