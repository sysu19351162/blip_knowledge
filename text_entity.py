import spacy
from tqdm import tqdm
import json
import re

#####读取数据###########
# print("ccs_synthetic_filtered_large")
# path='/data1/yangzhenbang_new/datasets/blip_caption/ccs_synthetic_filtered_large.json'

# print("ccs_synthetic_filtered")
# path='/data1/yangzhenbang_new/datasets/blip_caption/ccs_synthetic_filtered.json'

print("ccs_filtered")
path='/data1/yangzhenbang_new/datasets/blip_caption/ccs_filtered.json'

with open(path,'r',encoding='utf8')as fp:
    json_data = json.load(fp)
    # print(json_data)

nlp = spacy.load("en_core_web_sm")

entities = []
for idx,i in enumerate(tqdm(json_data)):
    # if idx ==5:
    #     break
    entity = []
    caption = i['caption']
    url = i['url']
    doc = nlp(caption)
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
    entities.append({'idx':idx,'entity':entity,'url':url})

# print(entities)
# res = dict(entities)
# with open("/data1/yangzhenbang_new/datasets/blip_caption/entities_ccs_synthetic_filtered_large.json", 'w', encoding='utf-8') as fw:
#     json.dump(entities, fw, indent=4)
# print("entities_ccs_synthetic_filtered_large")
# with open("/data1/yangzhenbang_new/datasets/blip_caption/entities_ccs_synthetic_filtered.json", 'w', encoding='utf-8') as fw:
#     json.dump(entities, fw, indent=4)
# print("entities_ccs_synthetic_filtered")


with open("/data1/yangzhenbang_new/datasets/blip_caption/entities_ccs_filtered.json", 'w', encoding='utf-8') as fw:
    json.dump(entities, fw, indent=4)
print("entities_ccs_filtered")

