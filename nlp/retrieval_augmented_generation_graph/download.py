#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir
from os.path import exists, join, splitext
import json
from wget import download
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_json', default = None, help = 'path to json')
  flags.DEFINE_string('output_dir', default = 'pdfs', help = 'path to directory')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  with open(FLAGS.input_json, 'r') as f:
    all_pdfs = json.loads(f.read())
  with sync_playwright() as p:
    browser = p.chromium.launch()
    for categories, pdfs in all_pdfs.items():
      for key, url in pdfs.items():
        parsed_url = urlparse(url)
        f = parsed_url.path.split('/')[-1]
        stem, ext = splitext(f)
        if ext in ['.htm', '.html']:
          download(url, out = join(FLAGS.output_dir, f))
        else:
          page = browser.new_page()
          page.goto(url)
          with page.expect_download() as download_info:
            download = download_info.value
          download.save_as(join('tmp', f))
          page.close()
    browser.close()

if __name__ == "__main__":
  add_options()
  app.run(main)
