#!/usr/bin/python3

allowed_nodes = ['electrolyte', 'conductivity', 'precursor']
rel_types = [
  ('electrolyte', 'has_conductivity', 'conductivity'),
  ('electrolyte', 'has_precursor', 'precursor')
]
