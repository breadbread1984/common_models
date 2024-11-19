# Introduction

this project demos on how to do facial recognition efficiently

# Usage

## Install prerequisites

```shell
python3 -m pip install -r requirements.txt
```

## sanity test

```shell
python3 db.py
```

to test whether faiss works properly.

## finetuning on target dataset

```shell
python3 create_dataset.py
python3 train.py
```

## test facial detection and recognition on CelebA

```shell
python3 recognition.py
```
