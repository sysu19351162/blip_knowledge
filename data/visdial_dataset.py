import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from models.blip import create_vit, init_tokenizer, load_checkpoint
from torchvision.datasets.utils import download_url


# {
#   'data': {
#     'questions': [
#       'does it have a doorknob',
#       'do you see a fence around the bear',
#       ...
#     ],
#     'answers': [
#       'no, there is just green field in foreground',
#       'countryside house',
#       ...
#     ],
#     'dialogs': [
#       {
#         'image_id': <image id>,
#         'caption': <image caption>,
#         'dialog': [
#           {
#             'question': <index of question in `data.questions` list>,
#             'answer': <index of answer in `data.answers` list>,
#             'answer_options': <100 candidate answer indices from `data.answers`>,
#             'gt_index': <index of `answer` in `answer_options`>
#           },
#           ... (10 rounds of dialog)
#         ]
#       },
#       ...
#     ]
#   },
#   'split': <VisDial split>,
#   'version': '1.0'
# }

class visdial_dataset(Dataset):
    def __init__(self, transform, img_root, train_files=[], split="train"):
        self.split = split
        self.ann_root = '/data1/yangzhenbang_new/datasets'
        self.transform = transform
        self.img_root = img_root
        self.rnd = 0
        self.cur_rnd = 0

        # if split == 'train':
        #     self.input = json.load(open(os.path.join(self.ann_root, 'visdial/train/visdial_1.0_train_processed_new.json'), 'r'))
        #     # self.dense_annotation = json.load(open(os.path.join(ann_root, 'visdial_1.0_train_processed.json'), 'r'))
        # elif split == 'test':
        #     self.input = json.load(open(os.path.join(self.ann_root, 'visdial/test/visdial_1.0_test_processed_new.json'), 'r'))
        #
        # else:
        #     self.input = json.load(open(os.path.join(self.ann_root, 'visdial/val/visdial_1.0_valid_processed_new.json'), 'r'))

        with open('/data1/yangzhenbang_new/datasets/visdial/train/visdial_1.0_train_processed_new.json') as f:
            self.visdial_data_train = json.load(f)

        with open('/data1/yangzhenbang_new/datasets/visdial/test/visdial_1.0_test_processed_new.json') as f:
            self.visdial_data_val = json.load(f)

        with open('/data1/yangzhenbang_new/datasets/visdial/val/visdial_1.0_val_processed_new.json') as f:
            self.visdial_data_test = json.load(f)

        self.overfit = False
        with open('/data1/visdial/visdial_1.0_val_dense_annotations_processed.json') as f:
            self.visdial_data_val_dense = json.load(f)
        self._split = 'train'
        self.subsets = ['train', 'val', 'test']
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = tokenizer
        self.tokenizer = init_tokenizer()
        self.image_path='/data1/yangzhenbang_new/datasets/visdial/'
        # fetching token indicecs of [CLS] and [SEP]
        tokens = ['[CLS]', '[MASK]', '[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.MASK = indexed_tokens[1]
        self.SEP = indexed_tokens[2]
        self._max_region_num = 37
        self.num_options = 100



    def __len__(self):
        return len(self.input['data']['dialogs'])

    def split(self):
        return self._split

    def __getitem__(self, index):

        def list2tensorpad(inp_list, max_seq_len):
            inp_tensor = torch.LongTensor([inp_list])
            inp_tensor_zeros = torch.zeros(1, max_seq_len, dtype=torch.long)
            inp_tensor_zeros[0, :inp_tensor.shape[1]] = inp_tensor
            inp_tensor = inp_tensor_zeros
            return inp_tensor

        def encode_input(utterances, start_segment, CLS, SEP, MASK, max_seq_len=256, max_sep_len=25, mask_prob=0.2):
            # output: the context, seperate context, the index of SEP
            cur_segment = start_segment
            token_id_list = []
            segment_id_list = []
            sep_token_indices = []
            masked_token_list = []

            token_id_list.append(CLS)
            segment_id_list.append(cur_segment)
            masked_token_list.append(0)

            cur_sep_token_index = 0

            for cur_utterance in utterances:
                # add the masked token and keep track
                # cur_masked_index = [1 if random.random() < mask_prob else 0 for _ in range(len(cur_utterance))]
                # masked_token_list.extend(cur_masked_index)
                token_id_list.extend(cur_utterance)
                segment_id_list.extend([cur_segment] * len(cur_utterance))

                token_id_list.append(SEP)
                segment_id_list.append(cur_segment)
                # masked_token_list.append(0)
                cur_sep_token_index = cur_sep_token_index + len(cur_utterance) + 1
                sep_token_indices.append(cur_sep_token_index)
                cur_segment = cur_segment ^ 1  # cur segment osciallates between 0 and 1

            assert len(segment_id_list) == len(token_id_list) == len(masked_token_list) == sep_token_indices[-1] + 1
            # convert to tensors and pad to maximum seq length
            tokens = list2tensorpad(token_id_list, max_seq_len)

            segment_id_list = list2tensorpad(segment_id_list, max_seq_len)
            # segment_id_list += 2
            # return tokens, segment_id_list, list2tensorpad(sep_token_indices, max_sep_len), masked_tokens
            return tokens, segment_id_list, list2tensorpad(sep_token_indices, max_sep_len)
        def tokens2str(seq):
            dialog_sequence = ''
            for sentence in seq:
                for word in sentence:
                    dialog_sequence += self.tokenizer._convert_id_to_token(word) + " "
                dialog_sequence += ' </end> '
            dialog_sequence = dialog_sequence.encode('utf8')
            return dialog_sequence

        def pruneRounds(context, num_rounds):
            start_segment = 1
            len_context = len(context)
            cur_rounds = (len(context) // 2) + 1
            l_index = 0
            if cur_rounds > num_rounds:
                # caption is not part of the final input
                l_index = len_context - (2 * num_rounds)
                start_segment = 0
            return context[l_index:], start_segment

        # Combining all the dialog rounds with the [SEP] and [CLS] token
        MAX_SEQ_LEN = 256
        cur_data = None
        if self._split == 'train':
            cur_data = self.visdial_data_train['data']
        elif self._split == 'val':
            if self.overfit:
                cur_data = self.visdial_data_train['data']
            else:
                cur_data = self.visdial_data_val['data']
        else:
            cur_data = self.visdial_data_test['data']

        # number of options to score on
        num_options = self.num_options
        assert num_options > 1 and num_options <= 100

        dialog = cur_data['dialogs'][index]
        cur_questions = cur_data['questions']
        cur_answers = cur_data['answers']
        img_id = dialog['image_id']

        if self._split == 'train':
                utterances = []
                utterances_random = []
                tokenized_caption = self.tokenizer.encode(dialog['caption'])
                # write the caption in
                utterances.append([tokenized_caption])
                utterances_random.append([tokenized_caption])
                tot_len = len(tokenized_caption) + 2  # add a 1 for the CLS token as well as the sep tokens which follows the caption
                for rnd, utterance in enumerate(dialog['dialog']):
                    # dialog history
                    cur_rnd_utterance = utterances[-1].copy()
                    cur_rnd_utterance_random = utterances[-1].copy()

                    tokenized_question = self.tokenizer.encode(cur_questions[utterance['question']])
                    tokenized_answer = self.tokenizer.encode(cur_answers[utterance['answer']])
                    #add QA pair
                    cur_rnd_utterance.append(tokenized_question)
                    cur_rnd_utterance.append(tokenized_answer)

                    question_len = len(tokenized_question)
                    answer_len = len(tokenized_answer)
                    tot_len += question_len + 1  # the additional 1 is for the sep token
                    tot_len += answer_len + 1  # the additional 1 is for the sep token

                    cur_rnd_utterance_random.append(self.tokenizer.encode(cur_questions[utterance['question']]))
                    # randomly select one random utterance in that round
                    utterances.append(cur_rnd_utterance)

                    num_inds = len(utterance['answer_options'])
                    gt_option_ind = utterance['gt_index']

                    negative_samples = []

                    # for _ in range(self.params["num_negative_samples"]):
                    for _ in range(1):
                        all_inds = list(range(100))
                        all_inds.remove(gt_option_ind)
                        all_inds = all_inds[:(num_options - 1)]
                        tokenized_random_utterance = None
                        option_ind = None

                        while len(all_inds):
                            option_ind = random.choice(all_inds)
                            tokenized_random_utterance = self.tokenizer.encode(
                                cur_answers[utterance['answer_options'][option_ind]])
                            # the 1 here is for the sep token at the end of each utterance
                            if (MAX_SEQ_LEN >= (tot_len + len(tokenized_random_utterance) + 1)):
                                break
                            else:
                                all_inds.remove(option_ind)
                        if len(all_inds) == 0:
                            # all the options exceed the max len. Truncate the last utterance in this case.
                            tokenized_random_utterance = tokenized_random_utterance[:answer_len]
                        t = cur_rnd_utterance_random.copy()
                        t.append(tokenized_random_utterance)
                        negative_samples.append(t)

                    utterances_random.append(negative_samples)
                # removing the caption in the beginning
                utterances = utterances[1:]
                utterances_random = utterances_random[1:]
                assert len(utterances) == len(utterances_random) == 10

                tokens_all_rnd = []
                mask_all_rnd = []
                segments_all_rnd = []
                sep_indices_all_rnd = []
                next_labels_all_rnd = []
                hist_len_all_rnd = []

                for j, context in enumerate(utterances):
                    tokens_all = []
                    mask_all = []
                    segments_all = []
                    sep_indices_all = []
                    next_labels_all = []
                    hist_len_all = []

                    # context, start_segment = pruneRounds(context, self.params['visdial_tot_rounds'])
                    context, start_segment = pruneRounds(context, 11)
                    # print("{}: {}".format(j, tokens2str(context)))
                    tokens, segments, sep_indices= encode_input(context, start_segment, self.CLS,
                                                                       self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN,
                                                                       mask_prob=self.params["mask_prob"])
                    tokens_all.append(tokens)
                    mask_all.append(mask)
                    sep_indices_all.append(sep_indices)
                    next_labels_all.append(torch.LongTensor([0]))
                    segments_all.append(segments)
                    hist_len_all.append(torch.LongTensor([len(context) - 1]))
                    negative_samples = utterances_random[j]

                    for context_random in negative_samples:
                        # context_random, start_segment = pruneRounds(context_random, self.params['visdial_tot_rounds'])
                        context_random, start_segment = pruneRounds(context_random, 11)
                        # print("{}: {}".format(j, tokens2str(context_random)))
                        tokens_random, segments_random, sep_indices_random = encode_input(context_random,
                                                                                                       start_segment,
                                                                                                       self.CLS,
                                                                                                       self.SEP,
                                                                                                       self.MASK,
                                                                                                       max_seq_len=MAX_SEQ_LEN,
                                                                                                       mask_prob=
                                                                                                       self.params[
                                                                                                           "mask_prob"])
                        tokens_all.append(tokens_random)
                        sep_indices_all.append(sep_indices_random)
                        next_labels_all.append(torch.LongTensor([1]))
                        segments_all.append(segments_random)
                        hist_len_all.append(torch.LongTensor([len(context_random) - 1]))

                    tokens_all_rnd.append(torch.cat(tokens_all, 0).unsqueeze(0))
                    segments_all_rnd.append(torch.cat(segments_all, 0).unsqueeze(0))
                    sep_indices_all_rnd.append(torch.cat(sep_indices_all, 0).unsqueeze(0))
                    next_labels_all_rnd.append(torch.cat(next_labels_all, 0).unsqueeze(0))
                    hist_len_all_rnd.append(torch.cat(hist_len_all, 0).unsqueeze(0))

                tokens_all_rnd = torch.cat(tokens_all_rnd, 0)
                segments_all_rnd = torch.cat(segments_all_rnd, 0)
                sep_indices_all_rnd = torch.cat(sep_indices_all_rnd, 0)
                next_labels_all_rnd = torch.cat(next_labels_all_rnd, 0)
                hist_len_all_rnd = torch.cat(hist_len_all_rnd, 0)

                item = {}
                item['captions'] = tokenized_caption
                item['tokens'] = tokens_all_rnd
                item['segments'] = segments_all_rnd
                item['sep_indices'] = sep_indices_all_rnd
                item['next_sentence_labels'] = next_labels_all_rnd
                item['hist_len'] = hist_len_all_rnd

                # get image
                image_path = os.join(self.image_path+'train/images', 'VisualDialog_train2018_'+str(img_id) + '.jpg')
                image = Image.open(image_path).convert('RGB')
                item['image'] = self.transform(image)

                return item

        elif self.split == 'val':
                # append all the 100 options and return all the 100 options concatenated with history
                # that will lead to 1000 forward passes for a single image
                tokenized_caption = self.tokenizer.encode(dialog['caption'])
                gt_relevance = None
                utterances = []
                gt_option_inds = []
                utterances.append([self.tokenizer.encode(dialog['caption'])])
                options_all = []
                for rnd, utterance in enumerate(dialog['dialog']):
                    cur_rnd_utterance = utterances[-1].copy()
                    cur_rnd_utterance.append(self.tokenizer.encode(cur_questions[utterance['question']]))
                    # current round
                    gt_option_ind = utterance['gt_index']
                    option_inds = []
                    option_inds.append(gt_option_ind)
                    all_inds = list(range(100))
                    all_inds.remove(gt_option_ind)
                    all_inds = all_inds[:(num_options - 1)]
                    option_inds.extend(all_inds)
                    gt_option_inds.append(0)
                    cur_rnd_options = []
                    answer_options = [utterance['answer_options'][k] for k in option_inds]
                    assert len(answer_options) == len(option_inds) == num_options
                    assert answer_options[0] == utterance['answer']

                    # if rnd == self.visdial_data_val_dense[index]['round_id'] - 1:
                    #     gt_relevance = torch.Tensor(self.visdial_data_val_dense[index]['gt_relevance'])
                    #     # shuffle based on new indices
                    #     gt_relevance = gt_relevance[torch.LongTensor(option_inds)]
                    for answer_option in answer_options:
                        cur_rnd_cur_option = cur_rnd_utterance.copy()
                        cur_rnd_cur_option.append(self.tokenizer.encode(cur_answers[answer_option]))
                        cur_rnd_options.append(cur_rnd_cur_option)
                    cur_rnd_utterance.append(self.tokenizer.encode(cur_answers[utterance['answer']]))
                    utterances.append(cur_rnd_utterance)
                    options_all.append(cur_rnd_options)
                # encode the input and create batch x 10 x 100 * max_len arrays (batch x num_rounds x num_options)
                tokens_all = []
                mask_all = []
                segments_all = []
                sep_indices_all = []
                hist_len_all = []

                for rnd, cur_rnd_options in enumerate(options_all):

                    tokens_all_rnd = []
                    # mask_all_rnd = []
                    segments_all_rnd = []
                    sep_indices_all_rnd = []
                    hist_len_all_rnd = []

                    for j, cur_rnd_option in enumerate(cur_rnd_options):
                        cur_rnd_option, start_segment = pruneRounds(cur_rnd_option, self.params['visdial_tot_rounds'])
                        tokens, segments, sep_indices = encode_input(cur_rnd_option, start_segment, self.CLS,
                                                                           self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN,
                                                                           mask_prob=0)

                        tokens_all_rnd.append(tokens)
                        # mask_all_rnd.append(mask)
                        segments_all_rnd.append(segments)
                        sep_indices_all_rnd.append(sep_indices)
                        hist_len_all_rnd.append(torch.LongTensor([len(cur_rnd_option) - 1]))

                    tokens_all.append(torch.cat(tokens_all_rnd, 0).unsqueeze(0))
                    # mask_all.append(torch.cat(mask_all_rnd, 0).unsqueeze(0))
                    segments_all.append(torch.cat(segments_all_rnd, 0).unsqueeze(0))
                    sep_indices_all.append(torch.cat(sep_indices_all_rnd, 0).unsqueeze(0))
                    hist_len_all.append(torch.cat(hist_len_all_rnd, 0).unsqueeze(0))

                tokens_all = torch.cat(tokens_all, 0)
                # mask_all = torch.cat(mask_all, 0)
                segments_all = torch.cat(segments_all, 0)
                sep_indices_all = torch.cat(sep_indices_all, 0)
                hist_len_all = torch.cat(hist_len_all, 0)

                item = {}
                item['captions'] = tokenized_caption
                item['tokens'] = tokens_all
                item['segments'] = segments_all
                item['sep_indices'] = sep_indices_all
                item['mask'] = mask_all
                item['hist_len'] = hist_len_all

                item['gt_option_inds'] = torch.LongTensor(gt_option_inds)

                # return dense annotation data as well
                # item['round_id'] = torch.LongTensor([self.visdial_data_val_dense[index]['round_id']])
                item['gt_relevance'] = gt_relevance

                #image
                image_path=os.join(self.image_path+'val/images','VisualDialog_test2018_'+str(img_id)+'.jpg')
                image = Image.open(image_path).convert('RGB')
                item['image'] = self.transform(image)
                # add image features. Expand them to create batch * num_rounds * num options * num bbox * img feats
                # features, num_boxes, boxes, _, image_target = self._image_features_reader[img_id]
                # features, spatials, image_mask, image_target, image_label = encode_image_input(features, num_boxes,
                #                                                                                boxes, \
                #                                                                                image_target,
                #                                                                                max_regions=self._max_region_num,
                #                                                                                mask_prob=0)
                #
                # item['image_feat'] = features
                # item['image_loc'] = spatials
                # item['image_mask'] = image_mask
                # item['image_target'] = image_target
                # item['image_label'] = image_label
                #
                # item['image_id'] = torch.LongTensor([img_id])

                return item

        else:
            assert num_options == 100
            cur_rnd_utterance = []
            options_all = []
            tokenized_caption = self.tokenizer.encode(dialog['caption'])
            gt_option_inds = []
            for rnd, utterance in enumerate(dialog['dialog']):
                cur_rnd_utterance.append(self.tokenizer.encode(cur_questions[utterance['question']]))
                if rnd != len(dialog['dialog']) - 1:
                    cur_rnd_utterance.append(self.tokenizer.encode(cur_answers[utterance['answer']]))
                gt_option_ind = utterance['gt_index']
                gt_option_inds.append(gt_option_ind)
            for answer_option in dialog['dialog'][-1]['answer_options']:
                cur_option = cur_rnd_utterance.copy()
                cur_option.append(self.tokenizer.encode(cur_answers[answer_option]))
                options_all.append(cur_option)

            tokens_all = []
            segments_all = []
            sep_indices_all = []
            hist_len_all = []


            for j, option in enumerate(options_all):
                option, start_segment = pruneRounds(option, self.params['visdial_tot_rounds'])
                print("option: {} {}".format(j, tokens2str(option)))
                tokens, segments, sep_indices = encode_input(option, start_segment, self.CLS,
                                                                       self.SEP, self.MASK, max_seq_len=MAX_SEQ_LEN,
                                                                       mask_prob=0)


                tokens_all.append(tokens)
                segments_all.append(segments)
                sep_indices_all.append(sep_indices)
                hist_len_all.append(torch.LongTensor([len(option) - 1]))

            tokens_all = torch.cat(tokens_all, 0)
            segments_all = torch.cat(segments_all, 0)
            sep_indices_all = torch.cat(sep_indices_all, 0)
            hist_len_all = torch.cat(hist_len_all, 0)

            item = {}
            item['captions'] = tokenized_caption
            item['tokens'] = tokens_all.unsqueeze(0)
            item['segments'] = segments_all.unsqueeze(0)
            item['sep_indices'] = sep_indices_all.unsqueeze(0)
            item['hist_len'] = hist_len_all.unsqueeze(0)
            item['gt_index'] = torch.LongTensor(gt_option_inds)
            item['image_id'] = torch.LongTensor([img_id])
            item['round_id'] = torch.LongTensor([dialog['round_id']])

            # image
            image_path = os.join(self.image_path+'test/images', 'VisualDialog_test2018_'+str(img_id) + '.jpg')
            image = Image.open(image_path).convert('RGB')
            item['image'] = self.transform(image)

            return item


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n