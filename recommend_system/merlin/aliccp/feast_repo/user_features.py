#!/usr/bin/python3

from datetime import timedelta
from feast import Entity, Field, FeatureView, ValueType
from feast.types import Int32
from feast.infra.offline_stores.file_source import FileSource

user_features = FileSource(
    path='data/user_features.parquet',
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
