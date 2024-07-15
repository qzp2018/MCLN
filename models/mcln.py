# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
from torch.nn import MultiheadAttention
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from .backbone_module import Pointnet2Backbone
from .modules import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead, PositionEmbeddingLearned
)
from .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)
from utils.scatter_util import scatter_mean
import pointnet2_utils
import einops
def calc_pairwise_locs(obj_centers,  eps=1e-10):

    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps) # (b, l, l)
    norm_pairwise_dists = pairwise_dists

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2]**2, 3)+eps)

    pairwise_locs = torch.stack(
        [norm_pairwise_dists, pairwise_locs[..., 2]/pairwise_dists, 
        pairwise_dists_2d/pairwise_dists, pairwise_locs[..., 1]/pairwise_dists_2d,
        pairwise_locs[..., 0]/pairwise_dists_2d],
        dim=3
    )
    return pairwise_locs


class SWA(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        
        query = self.with_pos_embed(query, pe)
        B = query.shape[1]
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[0], k.shape[0])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=None,attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output.transpose(0,1), output_weight, src_weight # (b, n_q, d_model), (b, n_q, n_v)

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output



class MCLN(nn.Module):
    """
    3D language grounder.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd
#-------------------text-head decoder-------------------------------------
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.swa_layers = nn.ModuleList([])
        self.swa_ffn_layers = nn.ModuleList([])
        self.rel_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 288))
        for i in range(3):
            self.swa_layers.append(SWA(d_model, nhead=8,dropout=0.2))
            self.swa_ffn_layers.append(FFN(d_model, hidden_dim=128, dropout=0.2, activation_fn='relu'))
        # Visual encoder
        self.backbone_net = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=1
        )
        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(torch.load(
                pointnet_ckpt
            ), strict=False)

        # Text Encoder
        # # (1) online
        # t_type = "roberta-base"
        # NOTE (2) offline: load from the local folder.
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                'data/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Mask Feats Generation layer
        self.x_mask = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.x_query = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.text_query_proj=nn.Sequential(
            nn.Linear(d_model,2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model,2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model,d_model)
            )
        self.super_grouper = pointnet2_utils.QueryAndGroup(radius=0.2, nsample=2, use_xyz=False, normalize_xyz=True)

        # Query initialization
        self.points_obj_cls = PointsObjClsModule(d_model)
        self.gsample_module = GeneralSamplingModule()
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_class, 1, num_queries, d_model,
            objectness=False, heading=False,
            compute_sem_scores=True
        )

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )

        # Init
        self.init_bn_momentum()
    
    
    # BRIEF visual and text backbones.
    def _run_backbones(self, inputs):
        """Run visual and text backbones."""
        # step 1. Visual encoder
        end_points = self.backbone_net(inputs['point_clouds'], end_points={}) # 50000 points -> 1024 points
        end_points['seed_inds'] = end_points['fp2_inds']                      # [batch_size,point_num]       
        end_points['seed_xyz'] = end_points['fp2_xyz']                        # [batch_size,point_num,3]
        end_points['seed_features'] = end_points['fp2_features']              # [batch_size,feature_dim,point_num]
        
        # step 2. Text encoder
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt"
        ).to(inputs['point_clouds'].device)
        
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)

        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized
        return end_points

    # BRIEF generate query.
    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)  # [B, 1, K=1024]
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        
        # top-k
        sample_inds = torch.topk(   
            torch.sigmoid(points_obj_cls_logits).squeeze(1),
            self.num_queries
        )[1].int()

        xyz, features, sample_inds = self.gsample_module(   
            xyz, features, sample_inds
        )

        end_points['query_points_xyz'] = xyz  # (B, V, 3)
        end_points['query_points_feature'] = features  # (B, F, V)
        end_points['query_points_sample_inds'] = sample_inds  # (B, V)
        return end_points
    
    # segmentation prediction
    def _seg_seeds_prediction(self, query, mask_feats, end_points, prefix=''):
        ## generate seed points masks
        pred_mask_seeds = torch.einsum('bnd,bdm->bnm', query, mask_feats)
        ## mapping seed points masks to superpoints masks
        end_points[f'{prefix}pred_mask_seeds'] = pred_mask_seeds
        return pred_mask_seeds

    def get_mask(self, query, mask_feats):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
        attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
        attn_masks = attn_masks.detach()

        return pred_masks, attn_masks

    def prediction_head(self, query, superpoint_feats):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, superpoint_feats)
        return pred_scores, pred_masks, attn_masks
    

    def avg_lang_feat(self, lang_feats, lang_masks):
        lang_len = lang_masks.sum(-1)
        lang_len = lang_len.unsqueeze(-1)
        lang_len[torch.where(lang_len == 0)] = 1
        return (lang_feats * ~lang_masks.unsqueeze(-1).expand_as(lang_feats)).sum(1) / lang_len

    # BRIEF forward.
    def forward(self, inputs):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B

                more keys if butd is enabled:
                    det_bbox_label_mask
                    det_boxes
                    det_class_ids
        Returns:
            end_points: dict
        """
        # STEP 1. vision and text encoding
        end_points = self._run_backbones(inputs)
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        text_feats = end_points['text_feats']
        text_padding_mask = end_points['text_attention_mask']
        end_points['coords'] = inputs['point_clouds'][..., 0:3].contiguous()
        
        # STEP 2. Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs['det_bbox_label_mask']

            # step box position.    det_boxes ([B, 132, 6]) -->  ([B, 128, 132])
            box_embeddings = self.box_embeddings(inputs['det_boxes'])
            # step box class        det_class_ids ([B, 132])  -->  ([B, 132, 160])
            class_embeddings = self.class_embeddings(self.butd_class_embeddings(inputs['det_class_ids']))
            # step box feature     ([B, 132, 288])
            detected_feats = torch.cat([box_embeddings, class_embeddings.transpose(1, 2)]
                                        , 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # STEP 3. Cross-modality encoding
        spatial_point_xyz=calc_pairwise_locs(points_xyz)
        points_features, text_feats = self.cross_encoder(
            vis_feats=points_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(
                len(points_xyz), points_xyz.size(1)
            ).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask,
            spatial_point_xyz=spatial_point_xyz
        )
        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()
        end_points["text_memory"] = text_feats
        end_points['seed_features'] = points_features
        
        # STEP 4. text projection --> 64
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens     # ([B, L, 64])

        # STEP 4.1 Mask Feats Generation
        mask_feats = self.x_mask(points_features)  # [B, 288, 1024]
        superpoint = inputs['superpoint']  # [B, 50000]
        end_points['superpoints'] = superpoint
        source_xzy = inputs['point_clouds'][..., 0:3].contiguous()  # [B, 50000, 3]
        super_features = []
        super_xyz_list = []
        for bs in range(source_xzy.shape[0]):
            super_xyz = scatter_mean(source_xzy[bs], superpoint[bs], dim=0).unsqueeze(0)  # [1, super_num, 3]  计算每个超点的平均坐标，即得到中心坐标
            super_xyz_list.append(super_xyz)
            grouped_feature,ball_idx = self.super_grouper(points_xyz[bs].unsqueeze(0), super_xyz, mask_feats[bs].unsqueeze(0))  # [1, 288, super_num, nsample]
            grouped_xyz =(points_xyz[bs])[ball_idx.long().squeeze(0)].unsqueeze(0)
            super_xyz_expand=super_xyz.unsqueeze(2)
            rel_coord=grouped_xyz-super_xyz_expand
            rel_feat=(self.rel_encoder(rel_coord)).permute(0,3,1,2)
            grouped_feature=grouped_feature+rel_feat     
            super_feature = F.max_pool2d(grouped_feature, kernel_size=[1, grouped_feature.size(3)]).squeeze(-1).squeeze(0)  # [288, super_num]
            super_features.append(super_feature)

        # STEP 5. Query Points Generation
        end_points = self._generate_queries(
            points_xyz, points_features, end_points
        )
        cluster_feature = end_points['query_points_feature']    # (B, F=288, V=256)
        cluster_xyz = end_points['query_points_xyz']            # (B, V=256, 3)
        query = self.decoder_query_proj(cluster_feature)        
        query = query.transpose(1, 2).contiguous()              # (B, V=256, F=288)
        # projection 288 --> 64
        if self.contrastive_align_loss: 
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )  # [B, 256, 64]

        # STEP 6.Proposals
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_'
        )
        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()
        query_mask = None
        query_last = None

        # STEP 7. Decoder
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers-1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # step Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None
            )  # (B, V, F)
            # step project
            if self.contrastive_align_loss:
                end_points[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # step box Prediction head
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),     # ([B, F=288, V=256])
                base_xyz=cluster_xyz,                   # ([B, 256, 3])
                end_points=end_points,  # 
                prefix=prefix
            )
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

            query_last = query

        # step Seg Prediction head
        query_last = self.x_query(query_last.transpose(1, 2)).transpose(1, 2)
#---------------------------text decoder-----------------------------------

        text_query=text_feats
        prediction_masks = []
        sp_pred_masks = []
        adaptive_weight_lists=[]

        for bs in range(query.shape[0]):
            bs_text_query=text_query[bs].unsqueeze(0)
            _, _, attn_masks = self.prediction_head(bs_text_query, super_features[bs].unsqueeze(0).transpose(1,2))
            for i in range(3): 
                bs_text_query, _,src_weight = self.swa_layers[i]( super_features[bs].unsqueeze(0).transpose(1,2).transpose(0,1),bs_text_query.transpose(0,1), attn_mask=attn_masks)#SWA模块
                bs_text_query = self.swa_ffn_layers[i](bs_text_query)
                _, _, attn_masks = self.prediction_head(bs_text_query, super_features[bs].unsqueeze(0).transpose(1,2))
            src_weight = src_weight.softmax(1)
            src_weight = torch.where(torch.isnan(src_weight), torch.zeros_like(src_weight), src_weight)#将src_weight中的NaN值替换为0
            q_score = (src_weight*~text_padding_mask[bs].unsqueeze(-1)).sum(-1) # [B, N_q]
            q_idx = q_score[0].argmax(dim=-1, keepdim=True) 
            pred_scores, pred_masks, _ = self.prediction_head(bs_text_query[:,q_idx,:], super_features[bs].unsqueeze(0).transpose(1,2))                
            prediction_masks.append(pred_masks.expand(1,256,pred_masks.shape[-1]))    # 将批次列表添加到总列表 
            adaptive_weight=torch.sigmoid(pred_scores.squeeze())
            adaptive_weight_lists.append(adaptive_weight)
            sp_pred_mask = self._seg_seeds_prediction(
                query_last[bs].unsqueeze(0),                                  # ([1, F=256, V=288])
                super_features[bs].unsqueeze(0),                             # ([1, F=288, V=super_num])
                end_points=end_points,  # 
                prefix=prefix
            ).squeeze(0)  
            sp_pred_masks.append(sp_pred_mask)
        end_points['sp_last_pred_masks'] = sp_pred_masks  # list  BS* [256, super_num]
        end_points['last_pred_masks'] = prediction_masks  # bs*[ 256, super_num]
        end_points['adaptive_weights']=adaptive_weight_lists
        end_points['super_xyz_list'] = super_xyz_list 

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
