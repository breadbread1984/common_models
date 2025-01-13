#!/usr/bin/python3

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class RAGTest(unittest.TestCase):
  def test_function(self):
    from models import Llama3_2
    from tools import load_rag
    llm = Llama3_2()
    tool = load_rag(llm)
    print('--------------------------------------------------------------------------')
    print(tool.invoke('What is the only one of the Seven Wonders of the Ancient World that still exists today?'))
    print(tool.invoke('What was the primary function of the Lighthouse of Alexandria?'))
    print(tool.invoke('Who was the sculptor responsible for creating the Statue of Zeus at Olympia?'))
    print('--------------------------------------------------------------------------')

if __name__ == "__main__":
  unittest.main()
