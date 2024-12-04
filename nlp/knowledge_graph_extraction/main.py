#!/usr/bin/python3

from absl import flags, app
from chains import load_graph_qa_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('tgi_host', default = 'http://localhost:8080/generate', help = 'host of TGI')
  flags.DEFINE_string('neo4j_host', default = 'bolt://localhost:7687', help = 'host')
  flags.DEFINE_string('neo4j_user', default = 'neo4j', help = 'user name')
  flags.DEFINE_string('neo4j_password', default = 'neo4j', help = 'password')
  flags.DEFINE_string('neo4j_db', default = 'neo4j', help = 'database')

def main(unused_argv):
  chain = load_graph_qa_chain(FLAGS.tgi_host, FLAGS.neo4j_host, FLAGS.neo4j_user, FLAGS.neo4j_password, FLAGS.neo4j_db)

