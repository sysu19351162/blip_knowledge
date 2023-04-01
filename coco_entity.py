import spacy
from tqdm import tqdm
import json
import re
###########data format###########
#idx, entity, caption, image_id


#####读取数据###########
print('COCO karpathy')
path='/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_train.json'
path1='/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_val.json'
path2='/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_test.json'

# with open(path,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)

with open(path1,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
#
# with open(path2,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)

# json_data.extend(json_data1)
# json_data.extend(json_data2)

nlp = spacy.load("en_core_web_sm")
index = []
entities = []
idx = 0
#train
for _,i in enumerate(tqdm(json_data)):
    # if idx == 5:
    #     break

    caption = (i['caption'])
    # print(type(i['caption']))
    # if(type(i['caption'])
    image_id = i['image']

    if type(caption) ==str:
        entity = []
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
        entities.append({'idx': idx, 'entity': entity, 'caption': caption, 'image_id': image_id})
        idx = idx+1
    else:
        for cap in caption:
            entity = []
            cap = cap.lower()
            doc = nlp(cap)
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
            entities.append({'idx': idx, 'entity': entity, 'caption': cap, 'image_id': image_id})
            idx = idx + 1

# print(entities)


# with open("/data1/yangzhenbang_new/datasets/blip_caption/coco_train_entity.json", 'w', encoding='utf-8') as fw:
#     json.dump(entities, fw, indent=4)
# print("coco_train_entity")

with open("/data1/yangzhenbang_new/datasets/blip_caption/data_knowledge/ccoco_val_entity.json", 'w', encoding='utf-8') as fw:
    json.dump(entities, fw, indent=4)
print("coco_val_entity")

# with open("/data1/yangzhenbang_new/datasets/blip_caption/data_knowledge/coco_test_entity.json", 'w', encoding='utf-8') as fw:
#     json.dump(entities, fw, indent=4)
# print("coco_test_entity")