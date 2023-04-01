import sys
import os
import clip
import torch
import argparse
import numpy as np
import json
import time
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
from data.utils import pre_caption
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import time

result_f = '/data1/yangzhenbang_new/datasets/blip_caption/coco_example_visual_concept.json'
with open(result_f) as f:
    result = json.load(f,strict=False)
d = {}
for i in result:
    print(type(i))
    d.update(i)

print(result)
#
print(type(result))
with open("/data1/yangzhenbang_new/datasets/blip_caption/example_visual_concept.json", 'w', encoding='utf-8') as fw:
        json.dump(d, fw, indent=4)


############################结果展示
# result_f = '/data1/yangzhenbang_new/datasets/blip_caption/coco_example_visual_concept1000.json'
# with open(result_f) as f:
#     result = json.load(f)
#
# for idx,i in enumerate(result):
#     if idx ==7:
#         break
#     print('filename:')
#     print(list(i.keys())[0])
#     r =  list(i.values())[0]
#     print("\nTop predictions:\n")
#     for j in r:
#         # print(value)
#         # print(index)
#         print(f"{list(j.keys())[0]:>16s}: {100 * list(j.values())[0]:.2f}%")