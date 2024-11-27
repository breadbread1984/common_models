#!/usr/bin/python3

from datetime import timedelta
from feast import Entity, Field, FeatureView, ValueType
from feast.types import Int32
from feast.infra.offline_stores.file_source import FileSource

def get_user_features(user_features_path):
  user_features = FileSource(
    path=user_features_path,
    timestamp_field="datetime",
    created_timestamp_column="created",
  )

  user = Entity(name="user_id", value_type=ValueType.INT32, join_keys=["user_id"],)

  user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(0),
    schema=[
        Field(name="user_shops", dtype=Int32),
        Field(name="user_profile", dtype=Int32),
        Field(name="user_group", dtype=Int32),
        Field(name="user_gender", dtype=Int32),
        Field(name="user_age", dtype=Int32),
        Field(name="user_consumption_2", dtype=Int32),
        Field(name="user_is_occupied", dtype=Int32),
        Field(name="user_geography", dtype=Int32),
        Field(name="user_intentions", dtype=Int32),
        Field(name="user_brands", dtype=Int32),
        Field(name="user_categories", dtype=Int32),
    ],
    online=True,
    source=user_features,
    tags=dict(),
  )
  return user_features_view

def get_item_features(item_features_path):
  item_features = FileSource(
    path=item_features_path,
    timestamp_field="datetime",
    created_timestamp_column="created",
  )

  item = Entity(name="item_id", value_type=ValueType.INT32, join_keys=["item_id"],)

  item_features_view = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(0),
    schema=[
        Field(name="item_category", dtype=Int32),
        Field(name="item_shop", dtype=Int32),
        Field(name="item_brand", dtype=Int32),
    ],
    online=True,
    source=item_features,
    tags=dict(),
  )
  return item_features_view

