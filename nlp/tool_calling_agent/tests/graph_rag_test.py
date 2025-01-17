#!/usr/bin/python3

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class GraphRAGTest(unittest.TestCase):
  def test_function(self):
    from models import Llama3_2
    from tools import load_graph_rag
    llm = Llama3_2()
    tool = load_graph_rag(llm)
    print(tool.invoke('Which organization does Elon Musk work as CEO of?'))
    print(tool.invoke('which organization develops Starship?'))

if __name__ == "__main__":
  unittest.main()
