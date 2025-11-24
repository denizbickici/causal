import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from collections import namedtuple
import random
import pickle

Batch_fuse = namedtuple('Batch_fuse', 'spatial_verb spatial_noun verb_label, noun_label, action_label logits_act, act_feat, verb_feat_cls, noun_feat_cls act_feat_cls')

        
class ActionFuseDataset(Dataset):
    def __init__(self,
                 root,
                 args=None,
                 is_val=False,
                 ):
        self.is_val = is_val
        self.data_root = root
        self.args = args
        
        """
        Here I load the extracted feats from Mvit and Lavila. Verb is Mvit trained on optical flow, Noun is Mvit trained on rgb. Act is lavila trained on action. You need to extract verb and noun feature from other modalities/pretrained Mvits as we discussed in the meeting.
        
        ek100
        """
        if is_val:
            verb_json_path = '/scratch/users/bickici/data/causal/verb_spatial/improved_63/egtea_test_feat.pt'
            noun_json_path = '/scratch/users/bickici/data/causal/noun/egtea_test_feat_1crop.pt'
            act_json_path = '/scratch/users/bickici/data/TIM/action_tokens_val/features/epic_val_feat.pt'
        else:
            verb_json_path = '/scratch/users/bickici/data/causal/verb_spatial/improved_63/egtea_train_feat.pt'
            noun_json_path = '/scratch/users/bickici/data/causal/noun/egtea_train_feat.pt'
            act_json_path = '/scratch/users/bickici/data/TIM/action_tokens_train/features/epic_train_feat.pt'
        
        """
        EGTEA
        if is_val:
            verb_json_path = '/home/dz/multi_modal_action_recognition/egtea_feat/mvit/verb/egtea_test_feat.pt'
            noun_json_path = '/home/dz/multi_modal_action_recognition/egtea_feat/mvit/noun/egtea_test_feat.pt'
            act_json_path = '/home/dz/multi_modal_action_recognition/egtea_feat/lavila/action/egtea_test_feat.pt'
        else:
            verb_json_path = '/home/dz/multi_modal_action_recognition/egtea_feat/mvit/verb/egtea_train_feat.pt'
            noun_json_path = '/home/dz/multi_modal_action_recognition/egtea_feat/mvit/noun/egtea_train_feat.pt'
            act_json_path = '/home/dz/multi_modal_action_recognition/egtea_feat/lavila/action/egtea_train_feat.pt'
        """


        self.verb_data = torch.load(verb_json_path)
        self.noun_data = torch.load(noun_json_path)
        self.act_data = torch.load(act_json_path)
        print(self.act_data.keys(), len(self.act_data['cls_feats']), self.act_data['cls_feats'][-1].shape)

        """
        here is the cls token feat from lavila, it is not used, you can ignore it
        """
        act_cls_feat = []
        if not is_val:
            for i in range(len(self.act_data['cls_feats'])):
                for j in range(len(self.act_data['cls_feats'][i])):
                    act_cls_feat.append(self.act_data['cls_feats'][i][j])
        
            #print(len(act_cls_feat), act_cls_feat[0].shape)
            self.act_data['cls_feats'] = act_cls_feat
        
        #print(self.verb_data['cls_feats'][0].shape, len(self.verb_data['cls_feats']))
        #print(self.verb_data['feats'][0].shape, len(self.verb_data['feats']))

    def __getitem__(self, index):

        """
        for verb and noun, I extract the features and shape it to temporal size 8 and feat size 768, for act, I extract and shape it to temporal size 16 and feat size 1024.
        spatial_verb_feat_cls, spatial_noun_feat_cls and spatial_act_feat_cls are cls token features. They are not used, you can ignore them.
        """
        spatial_verb_feat = self.verb_data['feats'][index]
        spatial_noun_feat = self.noun_data['feats'][index]

        spatial_verb_feat_cls = self.verb_data['cls_feats'][index] 
        spatial_noun_feat_cls = self.noun_data['cls_feats'][index] 
        
        spatial_act_logits = self.act_data['outputs'][index] 
        spatial_act_feat = self.act_data['feats'][index]
        spatial_act_feat_cls = self.act_data['cls_feats'][index]
        
        verb_label = self.verb_data['targets'][index]
        noun_label = self.noun_data['targets'][index]
        action_label = self.act_data['targets'][index]

        batch = Batch_fuse(spatial_verb_feat.to(torch.float), spatial_noun_feat.to(torch.float), verb_label, noun_label, action_label, spatial_act_logits.to(torch.float), spatial_act_feat.to(torch.float), spatial_verb_feat_cls.to(torch.float), spatial_noun_feat_cls.to(torch.float), spatial_act_feat_cls.to(torch.float))
        return batch

    def __len__(self):
        return len(self.verb_data['targets'])
        
