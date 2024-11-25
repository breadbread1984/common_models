#!/usr/bin/python3

from merlin.datasets.synthetic import generate_data
from mermlin.models.utils.dataset import unique_rows_by_features
from merlin.schema.tags import Tags

def load_datasets():
  train_raw, valid_raw = generate_data('aliccp-raw', 100000, set_sizes = (0.7, 0.3))
  user_features = unique_rows_by_features(train_raw, Tags.USER, Tags.USER_ID).compute().reset_index(drop=True)

if __name__ == "__main__":
  load_datasets()
