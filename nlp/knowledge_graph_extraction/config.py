#!/usr/bin/python3

node_types = ['Person', 'Location', 'Organization', 'Product']
rel_types = [
  ('Person', 'CEO_of', 'Organization'),
  ('Organization', 'Based_In', 'Location'),
  ('Product', 'Launches_From', 'Location'),
  ('Organization', 'Develops', 'Product'),
  ('Organization', 'Collaborates_With', 'Organization'),
  ('Product', 'Associated_With', 'Organization')
]
