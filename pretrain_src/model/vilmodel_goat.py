# GOAT
import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLayer

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad
from typing import Optional

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
logger = logging.getLogger(__name__)


from .Bert_backbone import *

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds

class LanguageEncoderDo(nn.Module):
    # add intervention
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False
        
        # for action and object
        if self.config.do_back_txt:
            if self.config.z_cross_attn:
                self.z_direc_cross_attn = RobertaAttention(config)
                self.z_landm_cross_attn = RobertaAttention(config)
            self.z_txt_linear = nn.Linear(config.hidden_size,config.hidden_size)
            self.z_direct_linear = nn.Linear(config.hidden_size,config.hidden_size)
            self.z_landm_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.z_concat_layernorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.z_direct_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.z_landm_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if self.config.do_back_txt_type == 'type_2':
                self.z_direc_cross_attn = RobertaAttention(config)
                self.z_landm_cross_attn = RobertaAttention(config)
                self.txt_self_attn = RobertaAttention(config)
                self.instr_aug_linear = nn.Linear(config.hidden_size,1)
                self.instr_ori_linear = nn.Linear(config.hidden_size,1)
                self.instr_sigmoid = nn.Sigmoid()
                self.concat_linear = nn.Linear(config.hidden_size*3, config.hidden_size)
        
        # Frontdoor Intervention
        if config.do_front_txt:
            self.z_front_cross_attn = RobertaAttention(config)
            self.z_front_linear = nn.Linear(config.hidden_size,config.hidden_size)
            self.z_front_ln = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, txt_embeds, txt_masks, z_direc_embeds=None, z_direc_pzs=None, z_landm_embeds=None, z_landm_pzs=None,
                front_txt_embeds=None):
        extended_txt_masks = extend_neg_masks(txt_masks)

        # BERT
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
 
        # Causal Intervention
        if z_direc_embeds is not None:
            if self.config.do_back_txt_type == 'type_1':
                if self.config.z_cross_attn:
                    z_direc_embeds = self.z_direc_cross_attn(z_direc_embeds, 
                                                             encoder_hidden_states=txt_embeds, 
                                                             encoder_attention_mask=extended_txt_masks)[0] 
                    z_landm_embeds = self.z_landm_cross_attn(z_landm_embeds, 
                                                             encoder_hidden_states=txt_embeds, 
                                                             encoder_attention_mask=extended_txt_masks)[0]
                p_z_direct = z_direc_embeds * z_direc_pzs.to(torch.float32) # [bs,len_d,dim]
                sum_z_direct = torch.sum(p_z_direct,1).unsqueeze(1) # [bs,1,dim]
                p_z_landm = z_landm_embeds * z_landm_pzs.to(torch.float32) # [bs,len_l,dim]
                sum_z_landm = torch.sum(p_z_landm,1).unsqueeze(1) # [bs,1,dim]
                txt_embeds = self.z_txt_linear(txt_embeds) + self.z_direct_linear(sum_z_direct) + self.z_landm_linear(sum_z_landm)
                if front_txt_embeds is not None:
                    # Frontdoor Intervention
                    z_front_embeds = self.z_front_cross_attn(txt_embeds, 
                                                            encoder_hidden_states=front_txt_embeds)[0] 
                    z_front_embeds = self.z_front_ln(self.z_front_linear(z_front_embeds))
                    txt_embeds = txt_embeds + z_front_embeds
                txt_embeds = self.z_concat_layernorm(txt_embeds)
            elif self.config.do_back_txt_type == 'type_2':
                z_direc_embeds = self.z_direc_cross_attn(txt_embeds, 
                                                        encoder_hidden_states=z_direc_embeds)[0] 
                z_direc_embeds = self.z_direct_ln(self.z_direct_linear(z_direc_embeds))
                if z_landm_embeds is not None:
                    z_landm_embeds = self.z_landm_cross_attn(txt_embeds, 
                                                            encoder_hidden_states=z_landm_embeds)[0]
                    z_landm_embeds = self.z_landm_ln(self.z_landm_linear(z_landm_embeds))
                
                if front_txt_embeds is not None:
                    # Frontdoor Intervention
                    z_front_embeds = self.z_front_cross_attn(txt_embeds, 
                                                            encoder_hidden_states=front_txt_embeds)[0] 
                    z_front_embeds = self.z_front_ln(self.z_front_linear(z_front_embeds))
                
                if self.config.do_add_method == 'door':
                    instr_aug_embeds = z_direc_embeds
                    if z_landm_embeds is not None:
                        instr_aug_embeds = instr_aug_embeds + z_landm_embeds
                    if front_txt_embeds is not None:
                        instr_aug_embeds = instr_aug_embeds + z_front_embeds
                    elif self.config.do_front_txt and front_txt_embeds is not None:
                        instr_aug_embeds = z_front_embeds
                        
                    aug_linear_weight = self.instr_aug_linear(instr_aug_embeds)
                    ori_linear_weight = self.instr_ori_linear(txt_embeds)
                    aug_weight = self.instr_sigmoid(aug_linear_weight+ori_linear_weight)
                    txt_embeds = torch.mul(aug_weight,instr_aug_embeds) + torch.mul((1-aug_weight),txt_embeds)  
                elif self.config.do_add_method == 'add':
                    txt_embeds = txt_embeds + z_direc_embeds + z_landm_embeds
                    if front_txt_embeds is not None:
                        txt_embeds = txt_embeds + z_front_embeds
                elif self.config.do_add_method == 'concat':
                    concat_txt_embeds = torch.cat((txt_embeds, z_direc_embeds, z_landm_embeds),-1)
                    txt_embeds = self.concat_linear(concat_txt_embeds)
                
                txt_embeds = self.z_concat_layernorm(txt_embeds)

        return txt_embeds

class CausalImageEmbeddings(nn.Module):
    ''' Causal learning & VGGT Feature Fusion & Scene Caption Fusion
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

        # =====================================================================
        # VGGT: attention-pooling + projection -> hidden fusion
        # =====================================================================
        self.vggt_dim = 2048

        self.vggt_proj = nn.Linear(self.vggt_dim, config.hidden_size)
        self.vggt_pool_score = nn.Linear(config.hidden_size, 1)
        self.vggt_post_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            BertLayerNorm(config.hidden_size, eps=1e-12),
            nn.Linear(config.hidden_size, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        
        dropout_prob = getattr(config, 'feat_dropout', config.hidden_dropout_prob)
        self.vggt_pool_dropout = nn.Dropout(dropout_prob)
        # =====================================================================

        # =====================================================================
        # [新增] Scene Caption Cross-Attention 融合层 
        # =====================================================================
        import copy
        tmp_config = copy.deepcopy(config)
        tmp_config.add_cross_attention = True
        self.caption_cross_att = BertAttention(tmp_config)
        self.caption_fusion_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # =====================================================================

        ''' For interventional image
        '''
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size+3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        if config.name != 'REVERIE' and config.name != 'SOON':
            self.img_self_attn = BertAttention(config)
            self.img_self_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
            
        self.do_back_img = config.do_back_img
        if self.do_back_img:
            self.do_img_before_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.do_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.do_img_attn = BertAttention(config)
            self.do_img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
            self.do_img_concat_layernorm = BertLayerNorm(config.hidden_size, eps=1e-12)

            if self.config.do_add_method == 'door':
                self.sigmoid = nn.Sigmoid()
            elif self.config.do_add_method == 'concat':
                self.do_concat_img_linear = nn.Linear(config.hidden_size*2, config.hidden_size)

        '''For reverie'''
        if self.config.name == 'REVERIE' or self.config.name == 'SOON':
            self.obj_name_linear = nn.Embedding(config.obj_name_vocab_size, config.hidden_size)
            self.obj_reverie_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_reverie_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.nav_type_embedding = nn.Embedding(3, config.hidden_size)
            self.pano_encoder = create_transformer_encoder(
                        config, config.num_pano_layers, norm=True
                    )
        else:
            self.nav_type_embedding = nn.Embedding(2, config.hidden_size)

        '''For global map aggregation'''
        if config.adaptive_pano_fusion: 
            self.adaptive_pano_attn = nn.Linear(config.hidden_size,1) 
            self.adaptive_pano_act = ACT2FN[config.hidden_act]
            self.adaptive_softmax = nn.Softmax(dim=1)

        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _fuse_vggt_into_view(self, view_img_embeds, traj_view_vggt_fts):
        x = traj_view_vggt_fts
        K = x.size(-2)
        D = x.size(-1)
        H = view_img_embeds.size(-1)

        x_flat = x.view(-1, D)
        x_proj = self.vggt_proj(x_flat)
        x_proj = x_proj.view(-1, K, H)

        scores = self.vggt_pool_score(x_proj).squeeze(-1)  
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1) 
        pooled = torch.sum(x_proj * attn, dim=-2)          
        
        pooled = self.vggt_pool_dropout(pooled)
        vggt_embeds = self.vggt_post_mlp(pooled)           

        vggt_embeds = vggt_embeds.view(view_img_embeds.shape)
        return view_img_embeds + vggt_embeds

    def forward(
        self, traj_view_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, type_embed_layer, 
        traj_reverie_obj_fts=None, traj_reverie_obj_lens=None,
        traj_reverie_obj_locs=None, 
        z_img_features=None, z_img_pzs=None, traj_reverie_obj_names=None,
        traj_view_vggt_fts=None, 
        scene_caption_embeds=None, 
        scene_caption_masks=None   
    ):
        view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))

        if self.config.name != 'REVERIE' and self.config.name != 'SOON':
            view_img_embeds = view_img_embeds + self.loc_layer_norm(self.loc_linear(traj_loc_fts))
        
        img_masks = gen_seq_masks(traj_vp_view_lens)
        extended_img_masks = extend_neg_masks(img_masks)

        if z_img_features is not None:
            z_img_embeds = self.do_img_layer_norm(self.do_img_before_linear(z_img_features))
            if self.config.z_cross_attn:
                z_img_embeds = self.do_img_attn(
                    z_img_embeds, 
                    encoder_hidden_states=view_img_embeds, encoder_attention_mask=extended_img_masks)[0]
            p_z_img = z_img_embeds * z_img_pzs.to(torch.float32)
            sum_z_img = torch.sum(p_z_img,1).unsqueeze(1) 
            view_img_embeds = self.img_after_linear(view_img_embeds) + self.do_img_after_linear(sum_z_img)
            view_img_embeds = self.do_img_concat_layernorm(view_img_embeds)

        # ================= [VGGT 注入点] =================
        # if traj_view_vggt_fts is not None:
        #     view_img_embeds = self._fuse_vggt_into_view(view_img_embeds, traj_view_vggt_fts)
        # =================================================

        # =====================================================================
        # [新增] 4.5 Scene Caption Cross-Attention 增强
        # =====================================================================
        if scene_caption_embeds is not None:
            S, V, H = view_img_embeds.size()
            L_cap = scene_caption_embeds.size(2) 

            # Query: 视觉特征 -> 变维为 (S*V, 1, H)
            visual_query = view_img_embeds.view(S*V, 1, H)
            
            # Context: 文本特征 -> (S*V, L_cap, H)
            caption_context = scene_caption_embeds.view(S*V, L_cap, H)
            
            # Mask: -> (S*V, 1, 1, L_cap)
            caption_masks = scene_caption_masks.view(S*V, 1, 1, L_cap) if scene_caption_masks is not None else None
            
            fused_outputs = self.caption_cross_att(
                hidden_states=visual_query,
                attention_mask=None,
                encoder_hidden_states=caption_context,
                encoder_attention_mask=caption_masks
            )
            
            # 还原形状 (S, V, H)
            fused_img = fused_outputs[0].view(S, V, H)
            view_img_embeds = self.caption_fusion_norm(view_img_embeds + fused_img)
        # =====================================================================

        if self.config.name != 'REVERIE' and self.config.name != 'SOON':
            view_img_embeds = self.dropout(view_img_embeds)
            view_img_embeds = self.img_self_encoder(
                view_img_embeds, src_key_padding_mask=img_masks.logical_not()
            )

        '''For REVERIE'''
        if traj_reverie_obj_fts is not None:
            reverie_obj_img_embeds = self.obj_reverie_linear(traj_reverie_obj_fts)
            if self.config.use_obj_name:
                if traj_reverie_obj_names is not None:
                    reverie_obj_img_embeds = reverie_obj_img_embeds + self.obj_name_linear(traj_reverie_obj_names)
            reverie_obj_img_embeds = self.obj_reverie_layer_norm(reverie_obj_img_embeds) 

            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                    view_img_embeds, reverie_obj_img_embeds, traj_vp_view_lens, traj_reverie_obj_lens
                ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            traj_vp_view_lens = traj_vp_view_lens + traj_reverie_obj_lens

            fused_img_reverie_embeds =  img_embeds +\
                    self.nav_type_embedding(traj_nav_types) +\
                    self.loc_layer_norm(self.loc_linear(traj_loc_fts))
            
            fused_img_reverie_embeds = self.layer_norm(fused_img_reverie_embeds)
            fused_img_reverie_embeds = self.dropout(fused_img_reverie_embeds)
                
            img_masks = gen_seq_masks(traj_vp_view_lens)
            view_img_embeds = self.pano_encoder(
                fused_img_reverie_embeds, src_key_padding_mask=img_masks.logical_not()
            )
        
        split_traj_embeds = torch.split(view_img_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_view_lens, traj_step_lens, 0)

        if self.config.adaptive_pano_fusion:
            traj_ori_embeds = view_img_embeds.clone()
            traj_fused_weight = self.adaptive_pano_attn(traj_ori_embeds)
            traj_fused_weight_act = torch.tanh(traj_fused_weight) 
            traj_fused_weight_act = self.adaptive_softmax(traj_fused_weight_act)
            traj_fused_embeded_update = torch.mul(traj_ori_embeds,traj_fused_weight_act)
            traj_fused_embeds = torch.sum(traj_fused_embeded_update,dim=1)
            split_traj_fused_embeds = torch.split(traj_fused_embeds, traj_step_lens, 0)
            return split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds
        
        return split_traj_embeds, split_traj_vp_lens, None


# class CausalImageEmbeddings(nn.Module):
#     ''' Causal learning & VGGT Feature Fusion
#     '''
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         # =====================================================================
#         # [修改: 两层 MLP + 加法融合 VGGT]
#         # =====================================================================
#         self.vggt_dim = 2048  
        
#         # 使用两层 MLP 充分处理 VGGT 特征，使其能与主干图像特征完美对齐
#         self.vggt_mlp = nn.Sequential(
#             nn.Linear(self.vggt_dim, config.hidden_size),
#             nn.GELU(), # 加入非线性激活，提升特征转换能力
#             BertLayerNorm(config.hidden_size, eps=1e-12),
#             nn.Linear(config.hidden_size, config.hidden_size),
#             BertLayerNorm(config.hidden_size, eps=1e-12)
#         )
#         # =====================================================================

#         ''' For interventional image
#         '''
#         self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
#         self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.loc_linear = nn.Linear(config.angle_feat_size+3, config.hidden_size)
#         self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
#         if config.name != 'REVERIE' and config.name != 'SOON':
#             self.img_self_attn = BertAttention(config)
#             self.img_self_encoder = create_transformer_encoder(
#                 config, config.num_pano_layers, norm=True
#             )
            
#         self.do_back_img = config.do_back_img
#         if self.do_back_img:
#             self.do_img_before_linear = nn.Linear(config.image_feat_size, config.hidden_size)
#             self.do_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
#             self.do_img_attn = BertAttention(config)
#             self.do_img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
#             self.img_after_linear = nn.Linear(config.hidden_size, config.hidden_size)
#             self.do_img_concat_layernorm = BertLayerNorm(config.hidden_size, eps=1e-12)

#             if self.config.do_add_method == 'door':
#                 self.sigmoid = nn.Sigmoid()
#             elif self.config.do_add_method == 'concat':
#                 self.do_concat_img_linear = nn.Linear(config.hidden_size*2, config.hidden_size)

#         '''For reverie'''
#         if self.config.name == 'REVERIE' or self.config.name == 'SOON':
#             self.obj_name_linear = nn.Embedding(config.obj_name_vocab_size, config.hidden_size)
#             self.obj_reverie_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
#             self.obj_reverie_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
#             self.nav_type_embedding = nn.Embedding(3, config.hidden_size)
#             self.pano_encoder = create_transformer_encoder(
#                         config, config.num_pano_layers, norm=True
#                     )
#         else:
#             self.nav_type_embedding = nn.Embedding(2, config.hidden_size)

#         '''For global map aggregation
#         '''
#         if config.adaptive_pano_fusion: 
#             self.adaptive_pano_attn = nn.Linear(config.hidden_size,1) 
#             self.adaptive_pano_act = ACT2FN[config.hidden_act]
#             self.adaptive_softmax = nn.Softmax(dim=1)

#         # 0: objects, 1: navigable
#         self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

        
#     def forward(
#         self, traj_view_img_fts, traj_loc_fts, traj_nav_types, 
#         traj_step_lens, traj_vp_view_lens, type_embed_layer, 
#         traj_reverie_obj_fts=None, traj_reverie_obj_lens=None,
#         traj_reverie_obj_locs=None, 
#         z_img_features=None, z_img_pzs=None, traj_reverie_obj_names=None,
#         traj_view_vggt_fts=None # <--- Recieve VGGT
#     ):
#         view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))

#         # =====================================================================
#         # [修改: 两层 MLP + 加法融合 VGGT] 严格保证空间结构不被破坏
#         # =====================================================================
#         if traj_view_vggt_fts is not None:
#             # 通过两层 MLP 将 VGGT 特征深度映射到与主干图像特征相同的子空间
#             # [Batch, ..., 36, 2048] -> [Batch, ..., 36, 768]
#             vggt_embeds = self.vggt_mlp(traj_view_vggt_fts)
            
#             # 使用加法进行残差式特征注入 (保持 View 一一对应)
#             view_img_embeds = view_img_embeds + vggt_embeds
#         # =====================================================================

#         if self.config.name != 'REVERIE' and self.config.name != 'SOON':
#             view_img_embeds = view_img_embeds + self.loc_layer_norm(self.loc_linear(traj_loc_fts))
        
#         img_masks = gen_seq_masks(traj_vp_view_lens)
#         extended_img_masks = extend_neg_masks(img_masks)

#         if z_img_features is not None:
#             # Do intervention
#             z_img_embeds = self.do_img_layer_norm(self.do_img_before_linear(z_img_features))
#             if self.config.z_cross_attn:
#                 z_img_embeds = self.do_img_attn(
#                     z_img_embeds, 
#                     encoder_hidden_states=view_img_embeds, encoder_attention_mask=extended_img_masks)[0]
#             p_z_img = z_img_embeds * z_img_pzs.to(torch.float32)
#             sum_z_img = torch.sum(p_z_img,1).unsqueeze(1) #[bs,1,dim]
#             view_img_embeds = self.img_after_linear(view_img_embeds) + self.do_img_after_linear(sum_z_img)
#             view_img_embeds = self.do_img_concat_layernorm(view_img_embeds)

#         if self.config.name != 'REVERIE' and self.config.name != 'SOON':
#             view_img_embeds = self.dropout(view_img_embeds)
#             view_img_embeds = self.img_self_encoder(
#                 view_img_embeds, src_key_padding_mask=img_masks.logical_not()
#             )

#         '''For REVERIE'''
#         if traj_reverie_obj_fts is not None:
#             reverie_obj_img_embeds = self.obj_reverie_linear(traj_reverie_obj_fts)
#             if self.config.use_obj_name:
#                 if traj_reverie_obj_names is not None:
#                     reverie_obj_img_embeds = reverie_obj_img_embeds + self.obj_name_linear(traj_reverie_obj_names)
#             reverie_obj_img_embeds = self.obj_reverie_layer_norm(reverie_obj_img_embeds) 

#             img_embeds = []
#             for view_embed, obj_embed, view_len, obj_len in zip(
#                     view_img_embeds, reverie_obj_img_embeds, traj_vp_view_lens, traj_reverie_obj_lens
#                 ):
#                 if obj_len > 0:
#                     img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
#                 else:
#                     img_embeds.append(view_embed[:view_len])
#             img_embeds = pad_tensors_wgrad(img_embeds)
#             traj_vp_view_lens = traj_vp_view_lens + traj_reverie_obj_lens

#             fused_img_reverie_embeds =  img_embeds +\
#                     self.nav_type_embedding(traj_nav_types) +\
#                     self.loc_layer_norm(self.loc_linear(traj_loc_fts))
            
#             fused_img_reverie_embeds = self.layer_norm(fused_img_reverie_embeds)
#             fused_img_reverie_embeds = self.dropout(fused_img_reverie_embeds)
                
#             img_masks = gen_seq_masks(traj_vp_view_lens)
#             view_img_embeds = self.pano_encoder(
#                 fused_img_reverie_embeds, src_key_padding_mask=img_masks.logical_not()
#             )
        
#         split_traj_embeds = torch.split(view_img_embeds, traj_step_lens, 0)
#         split_traj_vp_lens = torch.split(traj_vp_view_lens, traj_step_lens, 0)

#         if self.config.adaptive_pano_fusion:
#             traj_ori_embeds = view_img_embeds.clone()
#             traj_fused_weight = self.adaptive_pano_attn(traj_ori_embeds)
#             traj_fused_weight_act = torch.tanh(traj_fused_weight) 
#             traj_fused_weight_act = self.adaptive_softmax(traj_fused_weight_act)
#             traj_fused_embeded_update = torch.mul(traj_ori_embeds,traj_fused_weight_act)
#             traj_fused_embeds = torch.sum(traj_fused_embeded_update,dim=1)
#             split_traj_fused_embeds = torch.split(traj_fused_embeds, traj_step_lens, 0)
#             return split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds
        
#         return split_traj_embeds, split_traj_vp_lens, None

class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)
        if 'cfp' in config.pretrain_tasks:
            self.tim_self_encoder = BertAttention(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds]) # x[-1]: the current observation
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(vp_embeds, vp_masks, txt_embeds, txt_masks)
        return vp_embeds
    
    def forward_cfp(
        self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_masks = extend_neg_masks(vp_masks)
        vp_embeds = self.tim_self_encoder(vp_embeds, vp_masks)[0]
        return vp_embeds

class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)

        if 'cfp' in config.pretrain_tasks:
            self.tim_self_encoder = BertAttention(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        split_traj_fused_embeds=None
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                if split_traj_fused_embeds is not None:
                    visited_vp_fts[traj_vpids[i][t]] = split_traj_fused_embeds[i][t]
                else:
                    visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens,
        split_traj_fused_embeds=None
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            split_traj_fused_embeds=split_traj_fused_embeds
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None,
        split_traj_fused_embeds=None,
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, split_traj_fused_embeds=split_traj_fused_embeds,
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None
        
        gmap_embeds = self.encoder(
            gmap_embeds, gmap_masks, txt_embeds, txt_masks, 
            graph_sprels=graph_sprels
        )
        return gmap_embeds
    
    def forward_cfp(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None,
        split_traj_fused_embeds=None
    ):
        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, split_traj_fused_embeds=split_traj_fused_embeds
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_masks = extend_neg_masks(gmap_masks)
        gmap_self_embeds = self.tim_self_encoder(
            gmap_embeds, gmap_masks
        )[0]

        return gmap_self_embeds

class GlocalTextPathCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = RobertaEmbeddings(config)
        if config.do_back_txt:
            self.lang_encoder = LanguageEncoderDo(config)
        else:
            self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = CausalImageEmbeddings(config)
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)
        
        self.init_weights()

    # =================================================================
    # [修改] 辅助函数：将 Caption IDs 转换为文本 Embeddings (带防显存溢出的 Chunking)
    # =================================================================
    def _compute_caption_embeds(self, scene_caption_ids):
        if scene_caption_ids is None:
            return None, None
            
        S, V, L = scene_caption_ids.size()
        flat_ids = scene_caption_ids.view(-1, L)
        flat_masks = (flat_ids != 0).long()
        
        # 只做特征提取，切断梯度，避免爆显存
        with torch.no_grad():
            token_type_ids = torch.zeros_like(flat_ids)
            # 获取词嵌入
            txt_embeds = self.embeddings(flat_ids, token_type_ids=token_type_ids)
            if type(txt_embeds) is tuple:
                txt_embeds = txt_embeds[0]
            
            # --- 核心改动：Chunking (分批计算)，防止 Transformer 瞬间打爆显存 ---
            chunk_size = 512 # 每次最多让 512 个句���过 BERT，控制显存开销
            num_chunks = (flat_ids.size(0) + chunk_size - 1) // chunk_size
            
            out_embeds = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, flat_ids.size(0))
                
                chunk_embeds = txt_embeds[start_idx:end_idx]
                chunk_masks = flat_masks[start_idx:end_idx]
                
                if self.config.do_back_txt:
                    # 如果用了干预模块，这里取 layer[0] 作为特征 (参照你原本的代码)
                    res = self.lang_encoder.layer[0](chunk_embeds, extend_neg_masks(chunk_masks))[0]
                else:
                    res = self.lang_encoder(chunk_embeds, chunk_masks)
                    
                out_embeds.append(res)
                
            # 把分批计算的结果重新拼装起来
            txt_embeds = torch.cat(out_embeds, dim=0)
            # -----------------------------------------------------------
            
        extended_masks = extend_neg_masks(flat_masks)
        
        # 恢复维度：(Total_Steps, Max_Views, Cap_Len, Hidden)
        caption_embeds = txt_embeds.view(S, V, L, -1)
        caption_masks = extended_masks.view(S, V, 1, 1, L)
        return caption_embeds, caption_masks
    # =================================================================

    def forward(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        return_gmap_embeds=True,         
        z_img_features=None, z_img_pzs=None, traj_reverie_loc_fts=None, return_txt_embeds=False,
        traj_reverie_obj_names=None, 
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        traj_view_vggt_fts=None, # <--- [接收 VGGT 参数]
        scene_caption_ids=None   # <--- [新增参数]
    ):        
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_masks = gen_seq_masks(txt_lens)
        if self.config.do_back_txt:
            txt_embeds, z_direc_embeds,z_landm_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids, instr_z_direction_features=instr_z_direction_features, instr_z_landmark_features=instr_z_landmark_features)
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks, z_direc_embeds=z_direc_embeds, z_direc_pzs=instr_z_direction_pzs, z_landm_embeds=z_landm_embeds, z_landm_pzs=instr_z_landmark_pzs)
        else:
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)[0]
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        
        # ================= [新增] 处理 Caption IDs =================
        if scene_caption_ids is not None:
            # 直接移到 GPU 并计算，不再需要拆分重组
            scene_caption_ids = scene_caption_ids.to(txt_ids.device)
            cap_embeds, cap_masks = self._compute_caption_embeds(scene_caption_ids)
        else:
            cap_embeds, cap_masks = None, None
        # ========================================================

        # trajectory embedding
        # ================= [传递 Caption 与 VGGT] =================
        split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds = self.img_embeddings(
            traj_view_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, self.embeddings.token_type_embeddings,
            traj_obj_img_fts, traj_vp_obj_lens, traj_reverie_loc_fts,
            z_img_features, z_img_pzs, traj_reverie_obj_names,
            traj_view_vggt_fts=traj_view_vggt_fts,
            scene_caption_embeds=cap_embeds,
            scene_caption_masks=cap_masks
        )
        
        # gmap embeds
        if return_gmap_embeds:
            gmap_embeds = self.global_encoder(
                txt_embeds, txt_masks,
                split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
                gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=gmap_pair_dists,
                split_traj_fused_embeds=split_traj_fused_embeds
            )
        else:
            gmap_embeds = None

        # vp embeds
        vp_embeds = self.local_encoder(
            txt_embeds, txt_masks,
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )

        if return_txt_embeds:
            return txt_embeds, txt_masks, gmap_embeds, gmap_masks, vp_embeds, vp_masks, split_traj_fused_embeds
        else:
            return gmap_embeds, vp_embeds


    def forward_mlm(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        z_img_features=None, z_img_pzs=None, traj_reverie_loc_fts=None,
        traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        traj_view_vggt_fts=None, # <--- [接收 VGGT 参数]
        scene_caption_ids=None   # <--- [新增参数]
    ):
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_masks = gen_seq_masks(txt_lens)
        if self.config.do_back_txt:
            txt_embeds, z_direc_embeds,z_landm_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids, instr_z_direction_features=instr_z_direction_features, instr_z_landmark_features=instr_z_landmark_features)
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks, z_direc_embeds=z_direc_embeds, z_direc_pzs=instr_z_direction_pzs, z_landm_embeds=z_landm_embeds, z_landm_pzs=instr_z_landmark_pzs)
        else:
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)[0]
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks)

        # ================= [新增] 处理 Caption IDs =================
        if scene_caption_ids is not None:
            scene_caption_ids = scene_caption_ids.to(txt_ids.device)
            cap_embeds, cap_masks = self._compute_caption_embeds(scene_caption_ids)
        else:
            cap_embeds, cap_masks = None, None
        # ========================================================

        # trajectory embedding
        # ================= [传递 Caption 与 VGGT] =================
        split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds = self.img_embeddings(
            traj_view_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, self.embeddings.token_type_embeddings,
            traj_obj_img_fts, traj_vp_obj_lens, traj_reverie_loc_fts,
            z_img_features, z_img_pzs, traj_reverie_obj_names,
            traj_view_vggt_fts=traj_view_vggt_fts,
            scene_caption_embeds=cap_embeds,
            scene_caption_masks=cap_masks
        )

        # gmap embeds
        gmap_input_embeds, gmap_masks = self.global_encoder.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, split_traj_fused_embeds=split_traj_fused_embeds
        )
        gmap_txt_embeds = txt_embeds
        extended_gmap_masks = extend_neg_masks(gmap_masks)

        gmap_txt_embeds = self.global_encoder.encoder(
            gmap_txt_embeds, extend_neg_masks(txt_masks), # fixed to match self-attn input requirement
            gmap_input_embeds, extended_gmap_masks
        )

        # vp embeds
        vp_input_embeds, vp_masks = self.local_encoder.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_txt_embeds = txt_embeds
        extended_vp_masks = extend_neg_masks(vp_masks)
        vp_txt_embeds = self.local_encoder.encoder(
            vp_txt_embeds, extend_neg_masks(txt_masks), # fixed
            vp_input_embeds, extended_vp_masks,
        )

        txt_embeds = gmap_txt_embeds + vp_txt_embeds
        return txt_embeds

    def forward_cfp(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        return_gmap_embeds=True,         
        z_img_features=None, z_img_pzs=None, traj_reverie_loc_fts=None, return_txt_embeds=False,
        traj_reverie_obj_names=None,
        instr_z_landmark_features=None, instr_z_landmark_pzs=None,
        instr_z_direction_features=None, instr_z_direction_pzs=None,
        traj_view_vggt_fts=None, # <--- [接收 VGGT 参数]
        scene_caption_ids=None   # <--- [新增参数]
    ):        
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_masks = gen_seq_masks(txt_lens)
        if self.config.do_back_txt:
            txt_embeds, z_direc_embeds, z_landm_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids, instr_z_direction_features=instr_z_direction_features, instr_z_landmark_features=instr_z_landmark_features)
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks, z_direc_embeds=z_direc_embeds, z_direc_pzs=instr_z_direction_pzs, z_landm_embeds=z_landm_embeds, z_landm_pzs=instr_z_landmark_pzs)
        else:
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)[0]
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        
        # ================= [新增] 处理 Caption IDs =================
        if scene_caption_ids is not None:
            scene_caption_ids = scene_caption_ids.to(txt_ids.device)
            cap_embeds, cap_masks = self._compute_caption_embeds(scene_caption_ids)
        else:
            cap_embeds, cap_masks = None, None
        # ========================================================
        
        # trajectory embedding
        # ================= [传递 Caption 与 VGGT] =================
        split_traj_embeds, split_traj_vp_lens, split_traj_fused_embeds = self.img_embeddings(
            traj_view_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, self.embeddings.token_type_embeddings,
            traj_obj_img_fts, traj_vp_obj_lens, traj_reverie_loc_fts,
            z_img_features, z_img_pzs, traj_reverie_obj_names,
            traj_view_vggt_fts=traj_view_vggt_fts,
            scene_caption_embeds=cap_embeds,
            scene_caption_masks=cap_masks
        )
        
        # gmap embeds
        if return_gmap_embeds: 
            gmap_embeds = self.global_encoder.forward_cfp(
                split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
                gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=gmap_pair_dists,
                split_traj_fused_embeds=split_traj_fused_embeds
            )
        else:
            gmap_embeds = None

        # vp embeds
        vp_embeds = self.local_encoder.forward_cfp(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )

        if return_txt_embeds:
            return gmap_embeds, vp_embeds, txt_embeds
        else:
            return gmap_embeds, vp_embeds