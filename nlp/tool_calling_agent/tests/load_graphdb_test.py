#!/usr/bin/python3

import sys
import os
import unittest
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class LoadGraphDBTest(unittest.TestCase):
  def test_function(self):
    subprocess.call(['python3', os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'load_graphdb.py'), '--input_dir', os.path.join(os.path.abspath(os.path.dirname(__file__)), 'graphdb_test'), '--split'])

if __name__ == "__main__":
  unittest.main()
