from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer

import numpy as np


class BLIP_VISDIAL(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
                                                       drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        text_width = self.text_encoder.config.hidden_size
        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, image, text, context, labels=None, mode='train'):

        image_feats = self.visual_encoder(image)
        text.input_ids[:, 0] = self.tokenizer.enc_token_id

        context_output = self.text_encoder(context.input_ids,
                                           attention_mask=context.attention_mask,
                                           return_dict=True,
                                           mode='text'
                                           )

        if mode ==  'train':
            context_embeds = tile(context_output.last_hidden_state, 0,
                                  text.input_ids.shape[0] // context.input_ids.shape[0])
            context_atts = tile(context.attention_mask, 0, text.input_ids.shape[0] // context.input_ids.shape[0])

            image_embeds = tile(image_feats, 0, text.input_ids.shape[0] // image.shape[0])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=torch.cat([image_embeds, context_embeds], dim=1),
                                       encoder_attention_mask=torch.cat([image_atts, context_atts], dim=1),
                                       return_dict=True)
            scores = self.itm_head(output.last_hidden_state[:, 0, :])
            loss_itm = F.cross_entropy(scores, labels)

            return loss_itm

        else:
            image_embeds = tile(image_feats, 0, 100)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            context_embeds = tile(context_output.last_hidden_state, 0, 100)
            context_atts = tile(context.attention_mask, 0, 100)

            if mode == 'test':
                output = self.text_encoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=torch.cat([image_embeds, context_embeds], dim=1),
                                           encoder_attention_mask=torch.cat([image_atts, context_atts], dim=1),
                                           return_dict=True)
                features = output.last_hidden_state[:, 0, :]
            else:
                features = []
                for i in range(text.input_ids.shape[0] // 100):
                    output = self.text_encoder(text.input_ids[i * 100:i * 100 + 100],
                                               attention_mask=text.attention_mask[i * 100:i * 100 + 100],
                                               encoder_hidden_states=torch.cat([image_embeds, context_embeds],
                                                                               dim=1),
                                               encoder_attention_mask=torch.cat([image_atts, context_atts], dim=1),
                                               return_dict=True)
                    features.append(output.last_hidden_state[:, 0, :])

                features = torch.cat(features, dim=0)

            scores = self.itm_head(features)

            scores = scores[:, 1].view(-1, 100)
            inds = torch.argsort(scores, dim=1, descending=True)
            return inds






def blip_visdial(pretrained='', **kwargs):
    model = BLIP_VISDIAL(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    #         assert(len(msg.missing_keys)==0)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))