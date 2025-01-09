#!/usr/bin/python3

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestGraphRAGTest(unittest.TestCase):
  def test_function(self):
    from models import Llama3_2
    from tools import load_graph_rag
    llm = Llama3_2()
    tool = load_graph_rag(llm)
    print(tool.invoke('Where does Organization Tesla base in?'))

if __name__ == "__main__":
  unittest.main()
