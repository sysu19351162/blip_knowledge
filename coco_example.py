import spacy
from tqdm import tqdm
import json
import re

#####读取数据###########
path='/data1/yangzhenbang_new/datasets/coco/annotations/image_info_test2014.json'


with open(path,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    # json_data = json_data['annotations']
    # data=json_data[0:5]
    print(json_data)

nlp = spacy.load("en_core_web_sm")

phrases = []
for idx,i in enumerate(tqdm(data)):
    entity = []
    caption = i['caption']
    doc = nlp(caption)

    #使用NER
    # print(1)
    # for ent in doc.ents:
    #     print(ent.text, ent.label_)
    #     entity.append(ent.text)

    # 提取名词短语
    for chunk in doc.noun_chunks:
        # print ('{} - {}'.format(chunk,chunk.label_)) #注意chunk不是string，需要进行转换
        if ' and ' in str(chunk):
            list = re.split(r' and ', str(chunk))

            for i in list:
                doc1 = nlp(i)
                item = ''
                for w in doc1:
                    if((w.tag_ == 'NN' or w.tag_== 'NNS' or w.tag_== 'NNPS' or w.tag_== 'NNP')\
                            and w.text != 'the' and w.text != 'a'):
                        item = item +' '+ w.text
                if item != '' and item != ' ':
                    entity.append(item)
        else:
            item = ''
            for w in chunk:
                if ((w.tag_ == 'NN' or w.tag_ == 'NNS' or w.tag_ == 'NNPS' or w.tag_ == 'NNP') \
                        and w.text != 'the' and w.text != 'a'):
                    item = item + ' '+ w.text
            if item != '' and item != ' ':
                entity.append(item)
            # entity.append(str(chunk))

        # for phrase in entity:
        #     phrase = phrase.replace('the ', '')
        #     phrase = phrase.replace('a ', '')
        #     # print(phrase)
    #根据词性提取实体
    # for w in doc:
    #     if(w.tag_ == 'NN' or w.tag_ =='NNS'):
    #         entity.append(w.text)
    print(entity)
    phrases.append([idx,entity])



res = dict(phrases)
with open("/data1/yangzhenbang_new/datasets/blip_caption/coco_example.json", 'w', encoding='utf-8') as fw:
    json.dump(res, fw, indent=4)
