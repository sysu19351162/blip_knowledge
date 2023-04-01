import spacy
from tqdm import tqdm
import json
import re
# 1000092795.jpg
s=[]
# s.append('Two young guys with shaggy hair look at their hands while hanging out in the yard .')
# s.append('Two young , White males are outside near many bushes .')
# s.append('Two men in green shirts are standing in a yard .')
# s.append('A man in a blue shirt standing in a garden .')
# s.append('Two friends enjoy time spent together .')


# "val2014/COCO_val2014_000000535253.jpg"
s.append({'caption':"A variety of items is shown in a shopping cart.",'image':"val2014/COCO_val2014_000000535253.jpg"})
s.append({'caption':"A cart full of many types of different food.",'image':"val2014/COCO_val2014_000000535253.jpg"})
s.append({'caption':"A view from above looking into a shopping cart full of groceries",'image':"val2014/COCO_val2014_000000535253.jpg"})
s.append({'caption':"A shopping cart filled with lots of groceries.",'image':"val2014/COCO_val2014_000000535253.jpg"})
s.append({'caption':"A shopping cart full of food that includes bananas and milk.",'image':"val2014/COCO_val2014_000000535253.jpg"})

# "val2014/COCO_val2014_000000297147.jpg"
s.append({'caption':"A motorcycle parked on the pavement near a building.",'image':"val2014/COCO_val2014_000000297147.jpg"})
s.append({'caption':"A silver and black sports motorcycle parked near a building.",'image':"val2014/COCO_val2014_000000297147.jpg"})
s.append({'caption':"a black and silver motor cycle parked by a shed",'image':"val2014/COCO_val2014_000000297147.jpg"})
s.append({'caption':"A black and silver motorcycle outside on the road. ",'image':"val2014/COCO_val2014_000000297147.jpg"})
s.append({'caption':"This is a motorcycle that is parked near a building. ",'image':"val2014/COCO_val2014_000000297147.jpg"})

# "val2014/COCO_val2014_000000357586.jpg"
s.append({'caption':"A man with a small teddy bear peeking from a backpack.",'image':"val2014/COCO_val2014_000000357586.jpg"})
s.append({'caption':"A boy has a teddy bear in his backpack at church. ",'image':"val2014/COCO_val2014_000000357586.jpg"})
s.append({'caption':"A group of people sitting in a booth at a restaurant.",'image':"val2014/COCO_val2014_000000357586.jpg"})
s.append({'caption':"The boy has a teddy bear in his back pack. ",'image':"val2014/COCO_val2014_000000357586.jpg"})
s.append({'caption':"A man sitting on a bench with a teddy bear in his book bag",'image':"val2014/COCO_val2014_000000357586.jpg"})

# "val2014/COCO_val2014_000000339761.jpg"
s.append({'caption':"A bathroom with a white toilet next to a shower.",'image':"val2014/COCO_val2014_000000339761.jpg"})
s.append({'caption':"A bathroom with a large green plant growing on the wall.",'image':"val2014/COCO_val2014_000000339761.jpg"})
s.append({'caption':"A bathroom that has some plants growing in it.",'image':"val2014/COCO_val2014_000000339761.jpg"})
s.append({'caption': "A large wall of greenery is featured in a gray bathroom.",'image':"val2014/COCO_val2014_000000339761.jpg"})
s.append({'caption':"Leaves that are hanging on the side of a wall.",'image':"val2014/COCO_val2014_000000339761.jpg"})

# "val2014/COCO_val2014_000000218365.jpg"
s.append({'caption':"A metal vase filled with an orange flower.",'image':"val2014/COCO_val2014_000000218365.jpg"})
s.append({'caption':"An orchid blossom arranged in a silver cup.",'image':"val2014/COCO_val2014_000000218365.jpg"})
s.append({'caption':"A small steel vase with a flower of some sort in it.",'image':"val2014/COCO_val2014_000000218365.jpg"})
s.append({'caption':"An orange lily sits in a metal cup",'image':"val2014/COCO_val2014_000000218365.jpg"})
s.append({'caption':"A small metal bowl holding an orange flower on purple sheet.",'image':"val2014/COCO_val2014_000000218365.jpg"})

# "val2014/COCO_val2014_000000393225.jpg"
s.append({'caption':"A bowl of soup that has some carrots, shrimp, and noodles in it.",'image':"val2014/COCO_val2014_000000393225.jpg"})
s.append({'caption':"The healthy food is in the bowl and ready to eat. ",'image':"val2014/COCO_val2014_000000393225.jpg"})
s.append({'caption':"Soup has carrots and shrimp in it as it sits next to chopsticks.",'image':"val2014/COCO_val2014_000000393225.jpg"})
s.append({'caption':"A tasty bowl of ramen is served for someone to enjoy. ",'image':"val2014/COCO_val2014_000000393225.jpg"})
s.append({'caption':"Bowl of Asian noodle soup, with shrimp and carrots.",'image':"val2014/COCO_val2014_000000393225.jpg"})

# "val2014/COCO_val2014_000000262148.jpg"
s.append({'caption':"The skateboarder is putting on a show using the picnic table as his stage.",'image': "val2014/COCO_val2014_000000262148.jpg"})
s.append({'caption':"A skateboarder pulling tricks on top of a picnic table.",'image': "val2014/COCO_val2014_000000262148.jpg"})
s.append({'caption':"A man riding on a skateboard on top of a table.",'image': "val2014/COCO_val2014_000000262148.jpg"})
s.append({'caption':"A skate boarder doing a trick on a picnic table.",'image': "val2014/COCO_val2014_000000262148.jpg"})
s.append({'caption':"A person is riding a skateboard on a picnic table with a crowd watching.",'image': "val2014/COCO_val2014_000000262148.jpg"})


nlp = spacy.load("en_core_web_sm")

index = []
entities = []
idx = 0
for _,i in enumerate(tqdm(s)):
    # if idx == 5:
    #     break
    entity = []
    caption = (i['caption'])
    # print(type(i['caption']))
    # if(type(i['caption'])
    image_id = i['image']

    if type(caption) ==str:
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

# res = dict(entities)
# with open("/data1/yangzhenbang_new/datasets/blip_caption/entities_ccs_synthetic_filtered.json", 'w', encoding='utf-8') as fw:
#     json.dump(res, fw, indent=4)
# print(res)
with open("/data1/yangzhenbang_new/datasets/blip_caption/text_entity_example.json", 'w', encoding='utf-8') as fw:
    json.dump(entities, fw, indent=4)



