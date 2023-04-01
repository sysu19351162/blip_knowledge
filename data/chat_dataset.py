import os
import json

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

import spacy
nlp = spacy.load('en_core_web_sm')


class chat_coco_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=50, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        download_url(url, ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        pre_text = pre_caption(ann['caption'], self.max_words)  # TODO: check and modify max words?
        caption = self.prompt + pre_text

        # cjq
        doc = nlp(pre_text)
        noun_chunks = [nc for nc in doc.noun_chunks]
        noun_chunks = [n.text for n in noun_chunks]  # spacy class -> text
        # print('data: ', image.shape, caption, self.img_ids[ann['image_id']], noun_chunks)
        # cjq

        # image, caption, idx
        return image, caption, self.img_ids[ann['image_id']], noun_chunks


def train_collate_fn(batch):
    image_list, caption_list, idx_list, noun_list = [], [], [], []
    for image, caption, idx, noun_chunks in batch:
        image_list.append(image)
        caption_list.append(caption)
        idx_list.append(idx)
        noun_list.append(noun_chunks)

    # print('collate: ', torch.stack(image_list, dim=0).shape, caption_list, torch.Tensor(idx_list), noun_list)
    return torch.stack(image_list, dim=0), caption_list, torch.Tensor(idx_list), noun_list


# def test_collate_fn(batch):
#     image_list, caption_list, idx_list, noun_list = [], [], [], []
#     for image, caption, idx, noun_chunks in batch:
#         image_list.append(image)
#         caption_list.append(caption)
#         idx_list.append(idx)
#         noun_list.append(noun_chunks)
#
#     return torch.stack(image_list, dim=0), caption_list, torch.Tensor(idx_list), noun_list


class chat_coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=50):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}

        download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        # self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))[:512]  # TODO debug

        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index