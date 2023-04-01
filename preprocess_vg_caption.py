import json
from tqdm import tqdm

# result_f = '/data1/yangzhenbang_new/blip/BLIP/annotation/vg_caption.json'
# result_f = './annotation/vg_caption.json'
result_f = '/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_train.json'
with open(result_f) as f:
    result = json.load(f,strict=False)
d = {}
# for i in tqdm(result):
#     image = i['image']
#     head,seq,tail = image.partition('VG_100K')
#     i['image'] = '/data/datasets/vg/VG_100K'+tail
#     # i['image'] = '/data1/yangzhenbang_new/datasets/coco/' + image
#     d.update(i)
for i in tqdm(result):

    image = i['image']
    i['image'] = '/data1/yangzhenbang_new/datasets/coco/' + image
    i['img_name'] = i['image']
    d.update(i)

# print(result)
#
# print(type(result))
# with open("/data1/yangzhenbang_new/blip/BLIP/annotation/vg_caption.json", 'w',
#             encoding='utf-8') as fw:
#     json.dump(result, fw, indent=4)
with open("/data1/yangzhenbang_new/blip/BLIP/annotation/coco_karpathy_train_pretrain.json", 'w', encoding='utf-8') as fw:
        json.dump(result, fw, indent=4)