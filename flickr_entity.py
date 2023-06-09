import spacy
from tqdm import tqdm
import json
import re
import pandas as pd

###########data format###########
#idx, entity, caption, image_id



#####读取数据###########
print('flickr30k')
path='/data1/yangzhenbang_new/blip/BLIP/annotation/flickr30k_train.json'
path1='/data1/yangzhenbang_new/blip/BLIP/annotation/flickr30k_val.json'
path2='/data1/yangzhenbang_new/blip/BLIP/annotation/flickr30k_test.json'
with open(path,'r',encoding='utf8')as fp:
    json_data = json.load(fp)

with open(path1,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
#
with open(path2,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
#
# json_data.extend(json_data1)
# json_data.extend(json_data2)

nlp = spacy.load("en_core_web_sm")

index = []
entities1 = []
idx = 0
for _,i in enumerate(tqdm(json_data)):
    # if idx == 5:
    #     break
    # print(i)
    entity = []
    captions = i['caption']
    image_id = i['image']
    if type(captions) == str:
        caption = captions.lower()

        doc = nlp(caption)
        for chunk in doc.noun_chunks:
            # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
            if ' and ' in str(chunk):
                list = re.split(r' and ', str(chunk))

                for i in list:
                    doc1 = nlp(i)
                    item = ''
                    for w in doc1:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text

                    if item != '' and item != ' ':
                        entity.append(item)
            else:
                item = ''
                for w in chunk:
                    if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                            and w.text != 'the' and w.text != 'a'):
                        if item == '':
                            item = w.text
                        else:
                            item = item + ' ' + w.text
                if item != '' and item != ' ':
                    entity.append(item)
        # for w in doc:
        #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
        #         entity.append(w.text)
        # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
        # index.append(idx)
        # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
        entities1.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
        idx = idx + 1
    else:
        for caption in captions:
            caption = caption.lower()
            doc = nlp(caption)
            for chunk in doc.noun_chunks:
                # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
                if ' and ' in str(chunk):
                    list = re.split(r' and ', str(chunk))

                    for i in list:
                        doc1 = nlp(i)
                        item = ''
                        for w in doc1:
                            if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                    and w.text != 'the' and w.text != 'a'):
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text

                        if item != '' and item != ' ':
                            entity.append(item)
                else:
                    item = ''
                    for w in chunk:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text
                    if item != '' and item != ' ':
                        entity.append(item)
            # for w in doc:
            #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
            #         entity.append(w.text)
            # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
            # index.append(idx)
            # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
            entities1.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
            idx = idx + 1

with open("/data1/yangzhenbang_new/datasets/blip_caption/flickr_train_entity.json", 'w', encoding='utf-8') as fw:
    json.dump(entities1, fw, indent=4)
print("flickr_train_entity")

entities2 = []
idx = 0
for _,i in enumerate(tqdm(json_data1)):
    # if idx == 5:
    #     break
    # print(i)
    entity = []
    captions = i['caption']
    image_id = i['image']
    if type(captions) == str:
        caption = captions.lower()

        doc = nlp(caption)
        for chunk in doc.noun_chunks:
            # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
            if ' and ' in str(chunk):
                list = re.split(r' and ', str(chunk))

                for i in list:
                    doc1 = nlp(i)
                    item = ''
                    for w in doc1:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text

                    if item != '' and item != ' ':
                        entity.append(item)
            else:
                item = ''
                for w in chunk:
                    if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                            and w.text != 'the' and w.text != 'a'):
                        if item == '':
                            item = w.text
                        else:
                            item = item + ' ' + w.text
                if item != '' and item != ' ':
                    entity.append(item)
        # for w in doc:
        #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
        #         entity.append(w.text)
        # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
        # index.append(idx)
        # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
        entities2.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
        idx = idx + 1
    else:
        for caption in captions:
            caption = caption.lower()
            doc = nlp(caption)
            for chunk in doc.noun_chunks:
                # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
                if ' and ' in str(chunk):
                    list = re.split(r' and ', str(chunk))

                    for i in list:
                        doc1 = nlp(i)
                        item = ''
                        for w in doc1:
                            if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                    and w.text != 'the' and w.text != 'a'):
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text

                        if item != '' and item != ' ':
                            entity.append(item)
                else:
                    item = ''
                    for w in chunk:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text
                    if item != '' and item != ' ':
                        entity.append(item)
            # for w in doc:
            #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
            #         entity.append(w.text)
            # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
            # index.append(idx)
            # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
            entities2.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
            idx = idx + 1
with open("/data1/yangzhenbang_new/datasets/blip_caption/flickr_val_entity.json", 'w', encoding='utf-8') as fw:
    json.dump(entities2, fw, indent=4)
print("flickr_val_entity")

entities3 = []
idx = 0
for _,i in enumerate(tqdm(json_data2)):
    # if idx == 5:
    #     break
    # print(i)
    entity = []
    captions = i['caption']
    image_id = i['image']
    if type(captions) == str:
        caption = captions.lower()

        doc = nlp(caption)
        for chunk in doc.noun_chunks:
            # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
            if ' and ' in str(chunk):
                list = re.split(r' and ', str(chunk))

                for i in list:
                    doc1 = nlp(i)
                    item = ''
                    for w in doc1:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text

                    if item != '' and item != ' ':
                        entity.append(item)
            else:
                item = ''
                for w in chunk:
                    if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                            and w.text != 'the' and w.text != 'a'):
                        if item == '':
                            item = w.text
                        else:
                            item = item + ' ' + w.text
                if item != '' and item != ' ':
                    entity.append(item)
        # for w in doc:
        #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
        #         entity.append(w.text)
        # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
        # index.append(idx)
        # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
        entities3.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
        idx = idx + 1
    else:
        for caption in captions:
            caption = caption.lower()
            doc = nlp(caption)
            for chunk in doc.noun_chunks:
                # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
                if ' and ' in str(chunk):
                    list = re.split(r' and ', str(chunk))

                    for i in list:
                        doc1 = nlp(i)
                        item = ''
                        for w in doc1:
                            if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                    and w.text != 'the' and w.text != 'a'):
                                if item == '':
                                    item = w.text
                                else:
                                    item = item + ' ' + w.text

                        if item != '' and item != ' ':
                            entity.append(item)
                else:
                    item = ''
                    for w in chunk:
                        if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                                and w.text != 'the' and w.text != 'a'):
                            if item == '':
                                item = w.text
                            else:
                                item = item + ' ' + w.text
                    if item != '' and item != ' ':
                        entity.append(item)
            # for w in doc:
            #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
            #         entity.append(w.text)
            # entities.append([['idx',idx],['entity',entity],['caption',i],['image_id',image_id]]])
            # index.append(idx)
            # entities.append({'entity': entity, 'caption': i, 'image_id': image_id})
            entities3.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
            idx = idx + 1

with open("/data1/yangzhenbang_new/datasets/blip_caption/flickr_test_entity.json", 'w', encoding='utf-8') as fw:
    json.dump(entities3, fw, indent=4)
print("flickr_test_entity")
# print(entities)
# print(entities)
# res = dict(zip(index,entities))

# with open("/data1/yangzhenbang_new/datasets/blip_caption/flickr_entity.json", 'w', encoding='utf-8') as fw:
#     json.dump(entities, fw, indent=4)
# print("flickr_entity")



