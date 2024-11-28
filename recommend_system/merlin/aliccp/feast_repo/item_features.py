#!/usr/bin/python3

from datetime import timedelta
from feast import Entity, Field, FeatureView, ValueType
from feast.types import Int32
from feast.infra.offline_stores.file_source import FileSource

item_features = FileSource(
    path='data/item_features.parquet',
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

