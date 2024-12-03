#!/usr/bin/python3

allowed_nodes = ['person', 'place', 'organization', 'occupation', 'meeting']
rel_types = [
  ('person', 'work_for', 'organization'),
  ('person', 'lives_at', 'place'),
  ('person', 'job_is', 'occupation'),
  ('person', 'is_friend_of', 'person'),
  ('meeting', 'is_at', 'place')
]
