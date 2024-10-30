#!/usr/bin/python3

import os
import json
import xml.etree.ElementTree as ET

def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image_info = {
        "file_name": root.find('filename').text,
        "height": int(root.find('size/height').text),
        "width": int(root.find('size/width').text),
    }

    annotations = []
    for obj in root.findall('object'):
        category = obj.find('name').text
        bndbox = obj.find('bndbox')
        bbox = [
            int(bndbox.find('xmin').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
            int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
        ]

        annotations.append({
            "category": category,
            "bbox": bbox
        })

    return image_info, annotations

def voc_to_coco(voc_dir):
    categories = {}
    images = []
    annotations = []
    annotation_id = 0

    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith('.xml'):
            continue

        image_info, voc_annotations = parse_voc_annotation(os.path.join(voc_dir, xml_file))
        image_id = len(images)
        images.append({
            "id": image_id,
            "file_name": image_info["file_name"],
            "height": image_info["height"],
            "width": image_info["width"]
        })

        for voc_annotation in voc_annotations:
            category_name = voc_annotation["category"]
            if category_name not in categories:
                categories[category_name] = len(categories) + 1

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": categories[category_name],
                "bbox": voc_annotation["bbox"],
                "area": voc_annotation["bbox"][2] * voc_annotation["bbox"][3],
                "iscrowd": 0
            })
            annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": id, "name": name} for name, id in categories.items()]
    }

    return coco_format
