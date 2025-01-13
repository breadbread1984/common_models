# Introduction

this project implements RAG with langgraph.

# Usage

## Install prerequisite

```shell
python3 -m pip install requirements.txt
```

## Launch Neo4j

```shell
docker pull neo4j:enterprise-bullseye
docker run -d --publish=7474:7474 --publish=7687:7687 \
           --volume=$HOME/neo4j/data:/data \
           --name neo4j-apoc \
           -e NEO4J_apoc_export_file_enabled=true \
           -e NEO4J_apoc_import_file_enabled=true \
           -e NEO4J_apoc_import_file_use__neo4j__config=true \
           -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
           --privileged --shm-size 12G -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes --cpus=32 --memory=128G neo4j:enterprise-bullseye
```

create a database for receiving text chunks

## Load documents into vectordb

you can choose whatever documents and put them under a directory. after that, execute the following command

```shell
python3 load_text.py --input_dir <path/to/directory>
```

alternatively, you can use the demo text by executing the following command

```shell
python3 load_text.py --input_dir vectordb_test
```

## Launch the service

```shell
python3 main.py
```
