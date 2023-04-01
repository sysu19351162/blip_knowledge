import sys
import os
import numpy as np
import json
import time
from tqdm import tqdm
import time
import random

#entity_file_path = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/blip_caption/coco_entity.json"
entity_file_path = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/blip_caption/flickr_entity.json"


with open(entity_file_path) as f:
    entity = json.load(f)

knowledge_base_path_conceptnet = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/conceptnet_kg_object_as_key.json"

with open(knowledge_base_path_conceptnet) as f:
    knowledge_base_conceptnet = json.load(f)

knowledge_base_path_vg = "/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/vg_kg_object_as_key.json"

with open(knowledge_base_path_vg) as f:
    knowledge_base_vg = json.load(f)

threshold = 20

for i in tqdm(range(len(entity))):
#for i in tqdm(range(2)):
    entity_set = entity[i]['entity']
    #print('entity_set: ', entity_set)

    entity_knowledge_conceptnet = {}
    entity_overlap_knowledge_conceptnet = []
    entity_knowledge_vg = {}
    entity_overlap_knowledge_vg = []

    for entity_item in entity_set:
        for entity_key, knowledge_value in knowledge_base_conceptnet.items():
            if entity_item == entity_key:
                if len(knowledge_value) > threshold:
                    entity_knowledge_conceptnet[entity_item] = random.sample(knowledge_value, threshold)
                else:
                    entity_knowledge_conceptnet[entity_item] = knowledge_value
                for other_entity_item in entity_set:
                    if other_entity_item != entity_item:
                        for knowledge_item in knowledge_value:
                            triplet = knowledge_item.split('#')
                            if triplet[0] == entity_item and triplet[-1] == other_entity_item or triplet[0] == other_entity_item and triplet[-1] == entity_item:
                                entity_overlap_knowledge_conceptnet.append(knowledge_item)

                break

        for entity_key, knowledge_value in knowledge_base_vg.items():
            if entity_item == entity_key:
                if len(knowledge_value) > threshold:
                    entity_knowledge_vg[entity_item] = random.sample(knowledge_value, threshold)
                else:
                    entity_knowledge_vg[entity_item] = knowledge_value
                for other_entity_item in entity_set:
                    if other_entity_item != entity_item:
                        for knowledge_item in knowledge_value:
                            triplet = knowledge_item.split('#')
                            if triplet[0] == entity_item and triplet[-1] == other_entity_item or triplet[0] == other_entity_item and triplet[-1] == entity_item:
                                entity_overlap_knowledge_vg.append(knowledge_item)

                break


    entity[i]['knowledge_conceptnet'] = entity_knowledge_conceptnet
    entity[i]['knowledge_vg'] = entity_knowledge_vg

    entity[i]['overlap_knowledge_conceptnet'] = entity_overlap_knowledge_conceptnet
    entity[i]['overlap_knowledge_vg'] = entity_overlap_knowledge_vg
    #print('entity_knowledge: ', entity_knowledge)
    #print('overlap knowledge conceptnet', entity_overlap_knowledge_conceptnet)
    #print('overlap knowledge vg', entity_overlap_knowledge_vg)

#json.dump(entity, open("/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/coco_entity_has_knowledge.json", 'w'))
json.dump(entity, open("/mnt/data/linbingqian_new_1/project/LearnBeyondCap/kg_data/flickr_entity_has_knowledge.json", 'w'))


