#!/usr/bin/python3

import sys
import os
import unittest
from wget import download

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SQLRAGTest(unittest.TestCase):
  def test_function(self):
    from models import Llama3_2
    from tools import load_sql_rag
    from langchain import SQLDatabase
    llm = Llama3_2()
    tool = load_sql_rag(llm)
    sqlite_path = os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)),'Chinook_Sqlite.sqlite'))
    if not sqlite_path:
      download('https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite', out = os.path.abspath(os.path.dirname(__file__)))
    db = SQLDatabase.from_uri(f'sqlite:///{sqlite_path}')
    print(tool.invoke('List all albums released by AC/DC?'))
    print(tool.invoke('What is genre of music release by Santana?'))
    print(tool.invoke('List three most popular tracks among all play list?'))

if __name__ == "__main__":
  unittest.main()

