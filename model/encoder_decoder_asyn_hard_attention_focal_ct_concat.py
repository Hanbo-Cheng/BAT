import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import DenseNet
from .decoder_L2R_coverage_hard_attention_ct_concat_v3_hard_back import Decoder_L2R
from .decoder_R2L import Decoder_R2L 
import math
import copy
from .loss import *

# TODO: 待证实sigmoid focal loss要与softmax的优劣之后可能需要进一步修改
    
class Encoder_Decoder_Bi_Asyn(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder_Bi_Asyn, self).__init__()
        self.encoder = DenseNet(growthRate=params['growthRate'], 
                                reduction=params['reduction'],
                                bottleneck=params['bottleneck'], 
                                use_dropout=params['use_dropout'])
        self.init_context = nn.Linear(params['D'], params['n'])


        self.decoder_R2L = Decoder_R2L(params)
        self.decoder_L2R = Decoder_L2R(params)
        self.kl_criterion = torch.nn.KLDivLoss(reduction='batchmean',log_target=True).cuda()


        self.object_criterion_p1 = FocalLoss_mul()
        self.object_criterion_p2 = nn.CrossEntropyLoss(reduction='none')
        self.relation_criterion_p1 = FocalLoss_bi()
        self.relation_criterion_p2 = nn.BCEWithLogitsLoss(reduction='none')
        self.object_pix_criterion = nn.NLLLoss(reduction='none')

        # TODO： 视情况增加CAN中的counting 模块


    def forward(self, params, x, x_mask, C_y,  
                P_y, C_re, P_re, y_mask, re_mask, lp, rp, length):

        #encoder
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        #decoder

        C_y_L2R, C_y_R2L = C_y
        P_y_L2R, P_y_R2L = P_y
        P_re_L2R, P_re_R2L = P_re
        C_re_L2R, C_re_R2L= C_re
        y_mask_L2R, y_mask_R2L = y_mask
        re_mask_L2R, re_mask_R2L = re_mask
        lp_L2R, lp_R2L = lp
        rp_L2R, rp_R2L = rp

            # TODO： 更改deocder_R2L的输出，增加hidden state的返回，同时增加decoder_L2R的输入，增加hidden state的输入
        predict_objects_R2L, predict_relations_R2L, predict_objects_pix_R2L, R2L_hidden_states_ch, R2L_hidden_states_re = self.decoder_R2L(ctx, ctx_mask, 
                C_y_R2L, P_y_R2L, y_mask_R2L, P_re_R2L, init_state, length)
        predict_objects_L2R, predict_relations_L2R, predict_objects_pix_L2R, attention_hidden_ch = self.decoder_L2R(ctx, ctx_mask, 
                C_y_L2R, P_y_L2R, y_mask_L2R, P_re_L2R, init_state, length, R2L_hidden_states_ch, R2L_hidden_states_re,y_mask_R2L)
            
            
        
        max_len, batch, word_num = predict_objects_L2R.shape
        # compute loss 
        if(params['KL']):
            new_predict_objects_L2R = predict_objects_L2R.permute(1,0,2).reshape(-1, predict_objects_L2R.shape[2]) 
            new_predict_objects_R2L = predict_objects_R2L.permute(1,0,2).reshape(-1, predict_objects_L2R.shape[2]) 

        

        # hidden state 处理
        attention_hidden_ch = attention_hidden_ch.permute(1,0,2).reshape(-1, attention_hidden_ch.shape[-1])
        # L2R
        # predict_object_L2R : [len, batch, wordnum]
        # lp, rp 索引项 ，thinking label 不参与KL loss运算
        predict_objects_L2R = predict_objects_L2R.view(-1, predict_objects_L2R.shape[2])  # [len*batch, wordbnum]
        object_loss_L2R_p1 = self.object_criterion_p1(predict_objects_L2R, C_y_L2R.view(-1))
        

        object_loss_L2R_p2 = self.object_criterion_p2(predict_objects_L2R, C_y_L2R.view(-1))
        object_loss_L2R_p2 = object_loss_L2R_p2.view(C_y_L2R.shape[0], C_y_L2R.shape[1])
        object_loss_L2R_p2 = ((object_loss_L2R_p2 * y_mask_L2R).sum(0) / y_mask_L2R.sum(0)).mean()

        # object_loss_L2R = object_loss_L2R_p1 + object_loss_L2R_p2
        object_loss_L2R =  object_loss_L2R_p2
  

        predict_objects_pix_L2R = predict_objects_pix_L2R.view(-1, predict_objects_pix_L2R.shape[2])
        object_pix_loss_L2R = self.object_pix_criterion(predict_objects_pix_L2R, C_y_L2R.view(-1))
        object_pix_loss_L2R = object_pix_loss_L2R.view(C_y_L2R.shape[0], C_y_L2R.shape[1])
        object_pix_loss_L2R = ((object_pix_loss_L2R * y_mask_L2R).sum(0) / y_mask_L2R.sum(0)).mean()
            

        predict_relations_L2R_p1 = predict_relations_L2R.view(-1, predict_relations_L2R.shape[2])
        relation_loss_L2R_p1 = self.relation_criterion_p1(predict_relations_L2R_p1, C_re_L2R.view(-1, C_re_L2R.shape[2]))

        '''
        这些语句可能有误但不确定，所以先保留
        '''
        predict_relations_L2R_p2 = predict_relations_L2R.view(-1, predict_relations_L2R.shape[2])
        relation_loss_L2R_p2 = self.relation_criterion_p2(predict_relations_L2R_p2, C_re_L2R.view(-1, C_re_L2R.shape[2]))
        relation_loss_L2R_p2 = relation_loss_L2R_p2.view(C_re_L2R.shape[0], C_re_L2R.shape[1], C_re_L2R.shape[2])
        relation_loss_L2R_p2 = (relation_loss_L2R_p2 * re_mask_L2R[:, :, None]).sum(2).sum(0) / re_mask_L2R.sum(0)
        relation_loss_L2R_p2 = relation_loss_L2R_p2.mean()

        # relation_loss_L2R = relation_loss_L2R_p1 + relation_loss_L2R_p2
        relation_loss_L2R =  relation_loss_L2R_p2

        # R2L
        predict_objects_R2L = predict_objects_R2L.view(-1, predict_objects_R2L.shape[2])  # [len*batch, wordbnum]
        object_loss_R2L_p1 = self.object_criterion_p1(predict_objects_R2L, C_y_R2L.view(-1))

        '''
        这些语句可能有误但不确定，所以先保留
        '''
        object_loss_R2L_p2 = self.object_criterion_p2(predict_objects_R2L, C_y_R2L.view(-1))
        object_loss_R2L_p2 = object_loss_R2L_p2.view(C_y_R2L.shape[0], C_y_R2L.shape[1])
        object_loss_R2L_p2 = ((object_loss_R2L_p2 * y_mask_R2L).sum(0) / y_mask_R2L.sum(0)).mean()

        # object_loss_R2L  = object_loss_R2L_p1 + object_loss_R2L_p2
        object_loss_R2L  =  object_loss_R2L_p2

        predict_objects_pix_R2L = predict_objects_pix_R2L.view(-1, predict_objects_pix_R2L.shape[2])
        object_pix_loss_R2L = self.object_pix_criterion(predict_objects_pix_R2L, C_y_R2L.view(-1))
        object_pix_loss_R2L = object_pix_loss_R2L.view(C_y_R2L.shape[0], C_y_R2L.shape[1])
        object_pix_loss_R2L = ((object_pix_loss_R2L * y_mask_R2L).sum(0) / y_mask_R2L.sum(0)).mean()
            
        predict_relations_R2L_p1 = predict_relations_R2L.view(-1, predict_relations_R2L.shape[2])
        relation_loss_R2L_p1 = self.relation_criterion_p1(predict_relations_R2L_p1, C_re_R2L.view(-1, C_re_R2L.shape[2]))

        predict_relations_R2L_p2 = predict_relations_R2L.view(-1, predict_relations_R2L.shape[2])
        relation_loss_R2L_p2 = self.relation_criterion_p2(predict_relations_R2L_p2, C_re_R2L.view(-1, C_re_R2L.shape[2]))
        relation_loss_R2L_p2 = relation_loss_R2L_p2.view(C_re_R2L.shape[0], C_re_R2L.shape[1], C_re_R2L.shape[2])
        relation_loss_R2L_p2 = (relation_loss_R2L_p2 * re_mask_R2L[:, :, None]).sum(2).sum(0) / re_mask_R2L.sum(0)
        relation_loss_R2L_p2 = relation_loss_R2L_p2.mean()

        # relation_loss_R2L = relation_loss_R2L_p1 + relation_loss_R2L_p2
        relation_loss_R2L =  relation_loss_R2L_p2

        object_loss = object_loss_L2R + object_loss_R2L  
        relation_loss = relation_loss_L2R + relation_loss_R2L
        object_pix_loss = object_pix_loss_L2R + object_pix_loss_R2L

        lp_R2L = lp_R2L.permute(1,0)
        lp_L2R = lp_L2R.permute(1,0)

        '''
        原本兼容KL设置的，现在已经不需要
        '''
        # y_mask_R2L_new = lp_R2L.reshape(-1)
        # new_y_mask_nonzero_R2L = torch.nonzero(y_mask_R2L_new)  
        # y_mask_L2R_new = lp_L2R.reshape(-1)
        # new_y_mask_nonzero_L2R = torch.nonzero(y_mask_L2R_new)  
        # lp_R2L = torch.clamp(2*lp_R2L-1, 0, lp_R2L.shape[1])
        # lp_L2R = torch.clamp(2*lp_L2R-1, 0, lp_L2R.shape[1])
        # for i in range(batch):
        #     lp_R2L[i,:] += i*max_len
            # lp_L2R[i,:] += i*max_len

        # align_attn_score_L2R = torch.index_select(attention_hidden_ch, 0, lp_R2L.reshape(-1))
        # align_attn_score_L2R = torch.index_select(align_attn_score_L2R, 0, new_y_mask_nonzero_L2R.reshape(-1))  #感觉很可能是索引的问题
        # attn_label = torch.index_select(lp_L2R.reshape(-1), 0, new_y_mask_nonzero_L2R.reshape(-1))

        # attn_loss = self.object_criterion(align_attn_score_L2R, attn_label).mean()


        if(params['KL']):            
            pass
            # # TODO： 条件使用KL loss
            # new_predict_objects_R2L = torch.index_select(new_predict_objects_R2L, 0, new_y_mask_nonzero_R2L.reshape(-1))
            # KL_loss = self.kl_criterion(F.log_softmax(align_predict_objects_L2R/4,dim=-1),F.log_softmax(new_predict_objects_R2L/4,dim=-1))*16

            # loss = params['lc_lambda'] * object_loss + \
            #         params['lr_lambda'] * relation_loss + \
            #         params['lc_lambda_pix'] * object_pix_loss + \
            #         0.5 * KL_loss

            # return loss, object_loss, relation_loss, KL_loss
        else:

            loss = params['lc_lambda'] * object_loss + \
                    params['lr_lambda'] * relation_loss + \
                    params['lc_lambda_pix'] * object_pix_loss

            return loss, object_loss, relation_loss

        


    # TODO： 增加一个联合推理greedy_inference
    def greedy_inference(self, x, x_mask, max_length, p_y, p_re, p_mask):
        
        # 推理过程中需要修改这三个变量，因此需要做好备份
        p_mask_R2L = copy.deepcopy(p_mask)
        p_mask_L2R = copy.deepcopy(p_mask)
        p_y_R2L = copy.deepcopy(p_y)
        p_y_L2R = copy.deepcopy(p_y)
        p_re_R2L = copy.deepcopy(p_re)
        p_re_L2R = copy.deepcopy(p_re)
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        B, H, W = ctx_mask.shape
        
        # TODO： 同时使用两个方向的decoder 进行inference
        
        # R2L 推理过程
        attention_past_R2L = torch.zeros(B, 1, H, W).cuda()
        ctx_key_object_R2L = self.decoder_R2L.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation_R2L = self.decoder_R2L.conv_key_relation(ctx).permute(0, 2, 3, 1)
        
        relation_table_R2L = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static_R2L = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static_R2L = torch.zeros(B, max_length, 9)
        P_masks_R2L = torch.zeros(B, max_length).cuda()
        predict_childs_R2L = torch.zeros(max_length, B).to(torch.long).cuda()
        R2L_hidden_states_ch = torch.zeros(max_length, B, init_state.shape[-1]).cuda()
        R2L_hidden_states_re = torch.zeros(max_length, B, init_state.shape[-1]).cuda()


        ht = init_state
        ht_relation = init_state
        for i in range(max_length):
            predict_child, ht, attention, ct = self.decoder_R2L.get_child(ctx, ctx_key_object_R2L, ctx_mask, 
                attention_past_R2L, p_y_R2L, p_mask_R2L, p_re_R2L, ht)
            R2L_hidden_states_ch[i] = ht
            predict_childs_R2L[i]  = torch.argmax(predict_child, dim=1)
            predict_childs_R2L[i] *= p_mask_R2L.to(torch.long)
            attention_past_R2L = attention[:, None, :, :] + attention_past_R2L

            predict_relation, ht_relation = self.decoder_R2L.get_relation(ctx, ctx_key_relation_R2L, ctx_mask,
                predict_childs_R2L[i], ht_relation, ct) #(B, 9)
            R2L_hidden_states_re[i] = ht_relation
            P_masks_R2L[:, i] = p_mask_R2L 

            predict_relation_static_R2L[:, i, :] = predict_relation
            relation_table_R2L[:, i, :] = (predict_relation > 0)
            relation_table_static_R2L[:, i, :] = (predict_relation > 0)

            relation_table_R2L[:, :, 8] = relation_table_R2L[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask_R2L[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table_R2L[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table_R2L[ii, ji, iii] != 0:
                                p_re_R2L[ii] = iii
                                p_y_R2L[ii] = predict_childs_R2L[ji, ii]
                                relation_table_R2L[ii, ji, iii] = 0
                                relation_table_R2L[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask_R2L[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        
        # L2R推理过程
        P_masks_R2L = P_masks_R2L.permute(1,0)
        attention_past_L2R = torch.zeros(B, 1, H, W).cuda()
        r2l_maxlen = R2L_hidden_states_ch.shape[0]

        ctx_key_object_L2R = self.decoder_L2R.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation_L2R = self.decoder_L2R.conv_key_relation(ctx).permute(0, 2, 3, 1)

        R2L_hidden_key_ch = self.decoder_L2R.fc_key_R2L_hidden_ch(R2L_hidden_states_ch)
        R2L_hidden_key_re = self.decoder_L2R.fc_key_R2L_hidden_re(R2L_hidden_states_re)
        hidden_attention_past_ch = torch.zeros(B, 1, r2l_maxlen).cuda()
        hidden_attention_past_re = torch.zeros(B, 1, r2l_maxlen).cuda()
        
        relation_table_L2R = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static_L2R = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static_L2R = torch.zeros(B, max_length, 9)
        P_masks_L2R = torch.zeros(B, max_length).cuda()
        predict_childs_L2R = torch.zeros(max_length, B).to(torch.long).cuda()
        # L2R_hidden_states_ch = torch.zeros(max_length, B, init_state.shape[-1]).cuda()
        # L2R_hidden_states_re = torch.zeros(max_length, B, init_state.shape[-1]).cuda()

        ht = init_state
        ht_relation = init_state

        for i in range(max_length):
            predict_child, ht, attention, attention_hidden, ct = self.decoder_L2R.get_child(ctx, ctx_key_object_L2R, ctx_mask, 
                attention_past_L2R, hidden_attention_past_ch, p_y_L2R, p_mask_L2R, p_re_L2R, ht, R2L_hidden_states_ch, R2L_hidden_key_ch, P_masks_R2L)
            # L2R_hidden_states_ch[i] = ht
            predict_childs_L2R[i]  = torch.argmax(predict_child, dim=1)
            predict_childs_L2R[i] *= p_mask_L2R.to(torch.long)
            attention_past_L2R = attention[:, None, :, :] + attention_past_L2R
            hidden_attention_past_ch = attention_hidden[:, None, :] + hidden_attention_past_ch

            predict_relation, ht_relation, hidden_attention_re = self.decoder_L2R.get_relation(ctx, ctx_key_relation_L2R, ctx_mask,
                predict_childs_L2R[i], ht_relation, ct, R2L_hidden_states_re, R2L_hidden_key_re,P_masks_R2L, hidden_attention_past_re) #(B, 9)
            # L2R_hidden_states_re[i] = ht_relation
            hidden_attention_past_re = hidden_attention_re[:, None, :] + hidden_attention_past_re   
            P_masks_L2R[:, i] = p_mask_L2R

            predict_relation_static_L2R[:, i, :] = predict_relation
            relation_table_L2R[:, i, :] = (predict_relation > 0)
            relation_table_static_L2R[:, i, :] = (predict_relation > 0)

            relation_table_L2R[:, :, 8] = relation_table_L2R[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask_L2R[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table_L2R[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table_L2R[ii, ji, iii] != 0:
                                p_re_L2R[ii] = iii
                                p_y_L2R[ii] = predict_childs_L2R[ji, ii]
                                relation_table_L2R[ii, ji, iii] = 0
                                relation_table_L2R[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask_L2R[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        
        P_masks_R2L = P_masks_R2L.permute(1,0)
        
        
        # return predict_childs_R2L, P_masks_R2L, relation_table_static_R2L, predict_relation_static_R2L
        return predict_childs_L2R, P_masks_L2R, relation_table_static_L2R, predict_relation_static_L2R
    
    def greedy_inference_character_only(self, x, x_mask, max_length, p_y, p_re, c_re_L2R, p_mask):
        
        # 推理过程中需要修改这三个变量，因此需要做好备份
        # 仅对字符节点进行推理，故提供re的gt
        p_mask_R2L = copy.deepcopy(p_mask)
        p_mask_L2R = copy.deepcopy(p_mask)
        p_y_R2L = copy.deepcopy(p_y)
        p_y_L2R = copy.deepcopy(p_y)
        p_re_R2L = copy.deepcopy(p_re)
        p_re_L2R = copy.deepcopy(p_re)
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        B, H, W = ctx_mask.shape
        
        # TODO： 同时使用两个方向的decoder 进行inference
        
        # R2L 推理过程
        attention_past_R2L = torch.zeros(B, 1, H, W).cuda()
        ctx_key_object_R2L = self.decoder_R2L.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation_R2L = self.decoder_R2L.conv_key_relation(ctx).permute(0, 2, 3, 1)
        
        relation_table_R2L = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static_R2L = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static_R2L = torch.zeros(B, max_length, 9)
        P_masks_R2L = torch.zeros(B, max_length).cuda()
        predict_childs_R2L = torch.zeros(max_length, B).to(torch.long).cuda()
        R2L_hidden_states_ch = torch.zeros(max_length, B, init_state.shape[-1]).cuda()
        R2L_hidden_states_re = torch.zeros(max_length, B, init_state.shape[-1]).cuda()


        ht = init_state
        ht_relation = init_state
        for i in range(max_length):
            predict_child, ht, attention, ct = self.decoder_R2L.get_child(ctx, ctx_key_object_R2L, ctx_mask, 
                attention_past_R2L, p_y_R2L, p_mask_R2L, p_re_R2L, ht)
            R2L_hidden_states_ch[i] = ht
            predict_childs_R2L[i]  = torch.argmax(predict_child, dim=1)
            predict_childs_R2L[i] *= p_mask_R2L.to(torch.long)
            attention_past_R2L = attention[:, None, :, :] + attention_past_R2L

            predict_relation, ht_relation = self.decoder_R2L.get_relation(ctx, ctx_key_relation_R2L, ctx_mask,
                predict_childs_R2L[i], ht_relation, ct) #(B, 9)
            R2L_hidden_states_re[i] = ht_relation
            P_masks_R2L[:, i] = p_mask_R2L 

            predict_relation_static_R2L[:, i, :] = predict_relation
            relation_table_R2L[:, i, :] = (predict_relation > 0)
            relation_table_static_R2L[:, i, :] = (predict_relation > 0)

            relation_table_R2L[:, :, 8] = relation_table_R2L[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask_R2L[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table_R2L[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table_R2L[ii, ji, iii] != 0:
                                p_re_R2L[ii] = iii
                                p_y_R2L[ii] = predict_childs_R2L[ji, ii]
                                relation_table_R2L[ii, ji, iii] = 0
                                relation_table_R2L[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask_R2L[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        
        # L2R推理过程
        P_masks_R2L = P_masks_R2L.permute(1,0)
        attention_past_L2R = torch.zeros(B, 1, H, W).cuda()
        r2l_maxlen = R2L_hidden_states_ch.shape[0]

        ctx_key_object_L2R = self.decoder_L2R.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation_L2R = self.decoder_L2R.conv_key_relation(ctx).permute(0, 2, 3, 1)

        R2L_hidden_key_ch = self.decoder_L2R.fc_key_R2L_hidden_ch(R2L_hidden_states_ch)
        R2L_hidden_key_re = self.decoder_L2R.fc_key_R2L_hidden_re(R2L_hidden_states_re)
        hidden_attention_past_ch = torch.zeros(B, 1, r2l_maxlen).cuda()
        hidden_attention_past_re = torch.zeros(B, 1, r2l_maxlen).cuda()
        
        relation_table_L2R = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static_L2R = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static_L2R = torch.zeros(B, max_length, 9)
        P_masks_L2R = torch.zeros(B, max_length).cuda()
        predict_childs_L2R = torch.zeros(max_length, B).to(torch.long).cuda()
        # L2R_hidden_states_ch = torch.zeros(max_length, B, init_state.shape[-1]).cuda()
        # L2R_hidden_states_re = torch.zeros(max_length, B, init_state.shape[-1]).cuda()

        ht = init_state
        ht_relation = init_state

        for i in range(max_length):
            predict_child, ht, attention, attention_hidden, ct = self.decoder_L2R.get_child(ctx, ctx_key_object_L2R, ctx_mask, 
                attention_past_L2R, hidden_attention_past_ch, p_y_L2R, p_mask_L2R, p_re_L2R, ht, R2L_hidden_states_ch, R2L_hidden_key_ch, P_masks_R2L)
            # L2R_hidden_states_ch[i] = ht
            predict_childs_L2R[i]  = torch.argmax(predict_child, dim=1)
            predict_childs_L2R[i] *= p_mask_L2R.to(torch.long)
            attention_past_L2R = attention[:, None, :, :] + attention_past_L2R
            hidden_attention_past_ch = attention_hidden[:, None, :] + hidden_attention_past_ch

            predict_relation, ht_relation, hidden_attention_re = self.decoder_L2R.get_relation(ctx, ctx_key_relation_L2R, ctx_mask,
                predict_childs_L2R[i], ht_relation, ct, R2L_hidden_states_re, R2L_hidden_key_re,P_masks_R2L, hidden_attention_past_re) #(B, 9)
            # L2R_hidden_states_re[i] = ht_relation
            hidden_attention_past_re = hidden_attention_re[:, None, :] + hidden_attention_past_re   
            P_masks_L2R[:, i] = p_mask_L2R

            predict_relation = c_re_L2R[i]
            predict_relation_static_L2R[:, i, :] = predict_relation
            relation_table_L2R[:, i, :] = (predict_relation > 0)
            relation_table_static_L2R[:, i, :] = c_re_L2R[i,:,:]

            relation_table_L2R[:, :, 8] = relation_table_L2R[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask_L2R[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table_L2R[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table_L2R[ii, ji, iii] != 0:
                                p_re_L2R[ii] = iii
                                p_y_L2R[ii] = predict_childs_L2R[ji, ii]
                                relation_table_L2R[ii, ji, iii] = 0
                                relation_table_L2R[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask_L2R[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        
        P_masks_R2L = P_masks_R2L.permute(1,0)
        
        
        # return predict_childs_R2L, P_masks_R2L, relation_table_static_R2L, predict_relation_static_R2L
        return predict_childs_L2R, P_masks_L2R, relation_table_static_L2R, predict_relation_static_L2R
    
    def greedy_inference_relation_only(self, x, x_mask, max_length, p_y, p_re, c_y_L2R, p_mask):
        
        # 推理过程中需要修改这三个变量，因此需要做好备份
        # 仅预测关系分支，提供字符内容的gt
        p_mask_R2L = copy.deepcopy(p_mask)
        p_mask_L2R = copy.deepcopy(p_mask)
        p_y_R2L = copy.deepcopy(p_y)
        p_y_L2R = copy.deepcopy(p_y)
        p_re_R2L = copy.deepcopy(p_re)
        p_re_L2R = copy.deepcopy(p_re)
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        B, H, W = ctx_mask.shape
        
        # TODO： 同时使用两个方向的decoder 进行inference
        
        # R2L 推理过程
        attention_past_R2L = torch.zeros(B, 1, H, W).cuda()
        ctx_key_object_R2L = self.decoder_R2L.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation_R2L = self.decoder_R2L.conv_key_relation(ctx).permute(0, 2, 3, 1)
        
        relation_table_R2L = torch.zeros(B, max_length + 1, 9).to(torch.long)
        relation_table_static_R2L = torch.zeros(B,  max_length + 1, 9).to(torch.long)
        predict_relation_static_R2L = torch.zeros(B,  max_length + 1, 9)
        P_masks_R2L = torch.zeros(B,  max_length + 1).cuda()
        predict_childs_R2L = torch.zeros( max_length + 1, B).to(torch.long).cuda()
        R2L_hidden_states_ch = torch.zeros( max_length + 1, B, init_state.shape[-1]).cuda()
        R2L_hidden_states_re = torch.zeros( max_length + 1, B, init_state.shape[-1]).cuda()


        ht = init_state
        ht_relation = init_state
        for i in range(max_length + 1):
            predict_child, ht, attention, ct = self.decoder_R2L.get_child(ctx, ctx_key_object_R2L, ctx_mask, 
                attention_past_R2L, p_y_R2L, p_mask_R2L, p_re_R2L, ht)
            R2L_hidden_states_ch[i] = ht
            predict_childs_R2L[i]  = torch.argmax(predict_child, dim=1)
            predict_childs_R2L[i] *= p_mask_R2L.to(torch.long)
            attention_past_R2L = attention[:, None, :, :] + attention_past_R2L

            predict_relation, ht_relation = self.decoder_R2L.get_relation(ctx, ctx_key_relation_R2L, ctx_mask,
                predict_childs_R2L[i], ht_relation, ct) #(B, 9)
            R2L_hidden_states_re[i] = ht_relation
            P_masks_R2L[:, i] = p_mask_R2L 

            predict_relation_static_R2L[:, i, :] = predict_relation
            relation_table_R2L[:, i, :] = (predict_relation > 0)
            relation_table_static_R2L[:, i, :] = (predict_relation > 0)

            relation_table_R2L[:, :, 8] = relation_table_R2L[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask_R2L[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table_R2L[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table_R2L[ii, ji, iii] != 0:
                                p_re_R2L[ii] = iii
                                p_y_R2L[ii] = predict_childs_R2L[ji, ii]
                                relation_table_R2L[ii, ji, iii] = 0
                                relation_table_R2L[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask_R2L[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        
        # L2R推理过程
        P_masks_R2L = P_masks_R2L.permute(1,0)
        attention_past_L2R = torch.zeros(B, 1, H, W).cuda()
        r2l_maxlen = R2L_hidden_states_ch.shape[0]

        ctx_key_object_L2R = self.decoder_L2R.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation_L2R = self.decoder_L2R.conv_key_relation(ctx).permute(0, 2, 3, 1)

        R2L_hidden_key_ch = self.decoder_L2R.fc_key_R2L_hidden_ch(R2L_hidden_states_ch)
        R2L_hidden_key_re = self.decoder_L2R.fc_key_R2L_hidden_re(R2L_hidden_states_re)
        hidden_attention_past_ch = torch.zeros(B, 1, r2l_maxlen).cuda()
        hidden_attention_past_re = torch.zeros(B, 1, r2l_maxlen).cuda()
        
        relation_table_L2R = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static_L2R = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static_L2R = torch.zeros(B, max_length, 9)
        P_masks_L2R = torch.zeros(B, max_length).cuda()
        predict_childs_L2R = torch.zeros(max_length, B).to(torch.long).cuda()
        # L2R_hidden_states_ch = torch.zeros(max_length, B, init_state.shape[-1]).cuda()
        # L2R_hidden_states_re = torch.zeros(max_length, B, init_state.shape[-1]).cuda()

        ht = init_state
        ht_relation = init_state

        for i in range(max_length):
            predict_child, ht, attention, attention_hidden, ct = self.decoder_L2R.get_child(ctx, ctx_key_object_L2R, ctx_mask, 
                attention_past_L2R, hidden_attention_past_ch, p_y_L2R, p_mask_L2R, p_re_L2R, ht, R2L_hidden_states_ch, R2L_hidden_key_ch, P_masks_R2L)
            # L2R_hidden_states_ch[i] = ht
            predict_childs_L2R[i]  = c_y_L2R[i] # torch.argmax(predict_child, dim=1)
            predict_childs_L2R[i] *= p_mask_L2R.to(torch.long)
            attention_past_L2R = attention[:, None, :, :] + attention_past_L2R
            hidden_attention_past_ch = attention_hidden[:, None, :] + hidden_attention_past_ch

            predict_relation, ht_relation, hidden_attention_re = self.decoder_L2R.get_relation(ctx, ctx_key_relation_L2R, ctx_mask,
                predict_childs_L2R[i], ht_relation, ct, R2L_hidden_states_re, R2L_hidden_key_re, P_masks_R2L, hidden_attention_past_re) #(B, 9)
            # L2R_hidden_states_re[i] = ht_relation
            hidden_attention_past_re = hidden_attention_re[:, None, :] + hidden_attention_past_re   
            P_masks_L2R[:, i] = p_mask_L2R

            predict_relation_static_L2R[:, i, :] = predict_relation
            relation_table_L2R[:, i, :] = (predict_relation > 0)
            relation_table_static_L2R[:, i, :] = (predict_relation > 0)

            relation_table_L2R[:, :, 8] = relation_table_L2R[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask_L2R[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table_L2R[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table_L2R[ii, ji, iii] != 0:
                                p_re_L2R[ii] = iii
                                p_y_L2R[ii] = predict_childs_L2R[ji, ii]
                                relation_table_L2R[ii, ji, iii] = 0
                                relation_table_L2R[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask_L2R[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        
        P_masks_R2L = P_masks_R2L.permute(1,0)
        
        
        # return predict_childs_R2L, P_masks_R2L, relation_table_static_R2L, predict_relation_static_R2L
        return predict_childs_L2R, P_masks_L2R, relation_table_static_L2R, predict_relation_static_L2R    
    
    def greedy_inference_first_stage(self, x, x_mask, max_length, p_y, p_re, p_mask):
        
        # 推理过程中需要修改这三个变量，因此需要做好备份
        p_mask_R2L = copy.deepcopy(p_mask)
        p_mask_L2R = copy.deepcopy(p_mask)
        p_y_R2L = copy.deepcopy(p_y)
        p_y_L2R = copy.deepcopy(p_y)
        p_re_R2L = copy.deepcopy(p_re)
        p_re_L2R = copy.deepcopy(p_re)
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        B, H, W = ctx_mask.shape
        
        # TODO： 同时使用两个方向的decoder 进行inference
        
        # R2L 推理过程
        attention_past_R2L = torch.zeros(B, 1, H, W).cuda()
        ctx_key_object_R2L = self.decoder_R2L.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation_R2L = self.decoder_R2L.conv_key_relation(ctx).permute(0, 2, 3, 1)
        
        relation_table_R2L = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static_R2L = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static_R2L = torch.zeros(B, max_length, 9)
        P_masks_R2L = torch.zeros(B, max_length).cuda()
        predict_childs_R2L = torch.zeros(max_length, B).to(torch.long).cuda()
        R2L_hidden_states_ch = torch.zeros(max_length, B, init_state.shape[-1]).cuda()
        R2L_hidden_states_re = torch.zeros(max_length, B, init_state.shape[-1]).cuda()


        ht = init_state
        ht_relation = init_state
        for i in range(max_length):
            predict_child, ht, attention, ct = self.decoder_R2L.get_child(ctx, ctx_key_object_R2L, ctx_mask, 
                attention_past_R2L, p_y_R2L, p_mask_R2L, p_re_R2L, ht)
            R2L_hidden_states_ch[i] = ht
            predict_childs_R2L[i]  = torch.argmax(predict_child, dim=1)
            predict_childs_R2L[i] *= p_mask_R2L.to(torch.long)
            attention_past_R2L = attention[:, None, :, :] + attention_past_R2L

            predict_relation, ht_relation = self.decoder_R2L.get_relation(ctx, ctx_key_relation_R2L, ctx_mask,
                predict_childs_R2L[i], ht_relation, ct) #(B, 9)
            R2L_hidden_states_re[i] = ht_relation
            P_masks_R2L[:, i] = p_mask_R2L 

            predict_relation_static_R2L[:, i, :] = predict_relation
            relation_table_R2L[:, i, :] = (predict_relation > 0)
            relation_table_static_R2L[:, i, :] = (predict_relation > 0)

            relation_table_R2L[:, :, 8] = relation_table_R2L[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask_R2L[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table_R2L[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table_R2L[ii, ji, iii] != 0:
                                p_re_R2L[ii] = iii
                                p_y_R2L[ii] = predict_childs_R2L[ji, ii]
                                relation_table_R2L[ii, ji, iii] = 0
                                relation_table_R2L[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask_R2L[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
                
        return predict_childs_R2L, P_masks_R2L, relation_table_static_R2L, predict_relation_static_R2L

    def greedy_inference_stack(self, x, x_mask, max_length, p_y, p_re, p_mask):
        
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        ht = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        B, H, W = ctx_mask.shape
        attention_past = torch.zeros(B, 1, H, W).cuda()

        ctx_key = self.decoder.conv_key(ctx).permute(0, 2, 3, 1)
        
        parent_stack = [[] for Bi in range(B)]
        relation_stack = [[] for Bi in range(B)]
        relation_table_static = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static = torch.zeros(B, max_length, 9)
        P_masks = torch.zeros(B, max_length).cuda()
        predict_childs = torch.zeros(max_length, B).to(torch.long).cuda()

        
        for i in range(max_length):
            predict_child, ct, ht, attention = self.decoder.get_child(ctx, ctx_key, ctx_mask, 
                attention_past, p_y, p_mask, p_re, ht)
            predict_relation = self.decoder.get_relation(ct) #(B, 9)

            attention_past = attention[:, None, :, :] + attention_past
            P_masks[:, i] = p_mask 

            
            predict_childs[i]  = torch.argmax(predict_child, dim=1)
            predict_childs[i] *= p_mask.to(torch.long)

            predict_relation_static[:, i, :] = predict_relation
            relation_table_static[:, i, :] = (predict_relation > 0)

            #print(i, predict_childs[i], predict_relation)


            #greedy catch
            find_number = 0
            for Bi in range(B):
                if p_mask[Bi] < 0.5:
                    continue
                relation_stack[Bi].append(predict_relation[Bi] > 0)
                parent_stack[Bi].append(predict_childs[i, Bi])
                find_flag = 0
                while relation_stack[Bi] != []:
                    if relation_stack[Bi][-1][:8].sum() > 0:
                        for iii in range(9):
                            if relation_stack[Bi][-1][iii] != 0:
                                p_re[Bi] = iii
                                p_y[Bi] = parent_stack[Bi][-1]
                                relation_stack[Bi][-1][iii] = 0
                                if relation_stack[Bi][-1][:8].sum() == 0:
                                    relation_stack[Bi].pop()
                                    parent_stack[Bi].pop()
                                find_flag = 1
                                break
                    else:
                        relation_stack[Bi].pop()
                        parent_stack[Bi].pop()

                    if find_flag:
                        break
                
                find_number += find_flag
                if not find_flag:
                    p_mask[Bi] = 0.
            if find_number == 0:
                break
        return predict_childs, P_masks, relation_table_static, predict_relation_static

    def beamsearch_inference(self, x, x_mask, max_len, p_y, p_re, p_mask, beam_sise):
        # init state
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        ht = torch.tanh(self.init_context(ctx_mean))
        ctx_key_object = self.decoder.conv_key_object(ctx).permute(0, 2, 3, 1)
        ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation = self.decoder.conv_key_relation(ctx_pool).permute(0, 2, 3, 1)

        B, H, W = ctx_mask.shape
        attention_past = torch.zeros(B, 1, H, W).cuda()

        # decoding
 
        ctx_input = ctx.repeat(1, 1, 1, 1)
        ctx_input_pool = ctx_pool.repeat(1, 1, 1, 1)
        ctx_key_object_input = ctx_key_object.repeat(1, 1, 1, 1)
        ctx_key_relation_input = ctx_key_relation.repeat(1, 1, 1, 1)
        ctx_mask_input = ctx_mask.repeat(1, 1, 1)
        ctx_mask_input_pool = ctx_mask_pool.repeat(1, 1, 1)
        ht_input = ht
        y_input = p_y
        p_mask_input = p_mask.repeat(1)
        re_input = p_re

        # all we want
        labels = torch.zeros(beam_sise, max_len+1, dtype=torch.long).cuda()
        relation_tables = torch.zeros(beam_sise, max_len+1, 9, dtype=torch.long).cuda()
        len_labels = torch.zeros(beam_sise).cuda()
        logs = torch.zeros(beam_sise).cuda()
        labels_stack = [[] for bi in range(beam_sise)]
        relation_stack = [[] for bi in range(beam_sise)]
        # attention_list = [[] for i in range(beam_sise)]
        
        
        N = beam_sise
        end_N = 0
        for ei in range(max_len):
            score, ct, ht, attention = self.decoder.get_child(ctx_input, 
                ctx_key_object_input, ctx_mask_input, 
                attention_past, y_input, p_mask_input, re_input, ht_input)
            attention_past = attention[:, None, :, :] + attention_past


            # for visualization
            # alpha_np = alpha.cuda().numpy()
            
            # 复制当前没结束的前N项
            t_labels = copy.deepcopy(labels[:N, :])
            t_relation_tables = copy.deepcopy(relation_tables[:N, :, :])
            t_len_labs = copy.deepcopy(len_labels[:N])
            t_logs = copy.deepcopy(logs[:N])
            t_labels_stack = copy.deepcopy(labels_stack[:N])
            t_relation_stack = copy.deepcopy(relation_stack[:N])
            # t_alpha_list = copy.deepcopy(attention_list)

            #创建下次循环需要的变量：
            ys = torch.LongTensor(N).cuda()
            res = torch.LongTensor(N).cuda()
            hts = torch.Tensor(N, ht.shape[1]).cuda()
            atts = torch.Tensor(N, 1, H, W).cuda()

            # 计算出此次综合概率前N项
            log_prob_y = torch.log( F.softmax(score, 1) ) #(N,K)
            max_logs, max_ys = torch.topk(log_prob_y, N, 1) #(N,N) (N,N)
            if ei == 0:
                t_all_logs = max_logs
            else:
                t_all_logs = max_logs + t_logs[:,None] #(N,N)
            t_logs, t_max_indexs = torch.topk(t_all_logs.view(-1), N) #(N.) (N.)
            
            # 得到此次log最大的前N项的predict
            t_ys = torch.LongTensor(N).cuda()
            t_ct = torch.zeros(N, ct.shape[1]).cuda()
            t_ht = torch.zeros(N, ht.shape[1]).cuda()
            column_len = N
            for yi in range(column_len):
                index = t_max_indexs[yi].item()
                row = int(index/column_len)
                column = index%column_len
                t_ys[yi] = max_ys[row][column].item()
                t_ct[yi] = ct[row]
                t_ht[yi] = ht[row]

            predict_relation = self.decoder.get_relation(ctx_input_pool, 
                ctx_key_relation_input, ctx_mask_input_pool,
                t_ct, t_ys, t_ht)
            predict_relation = (predict_relation >= 0)
            # t_predict_relation = copy.deepcopy(predict_relation)

            t_end = 0 #本次label终止的个数
            column_len = N
            for yi in range(column_len):
                index = t_max_indexs[yi].item()
                row = int(index/column_len)
                column = index%column_len
                t_y = t_ys[yi]

                tt_relation_stack = copy.deepcopy(t_relation_stack[row])
                tt_label_stack = copy.deepcopy(t_labels_stack[row])
                tt_relation_stack.append(copy.deepcopy(predict_relation[row]))
                tt_label_stack.append(t_y)
                t_p_re, t_p = self.find_parent(tt_relation_stack, tt_label_stack)
                
                if t_p_re == 8:
                    end_N += 1
                    N = beam_sise - end_N
                    logs[N] = t_logs[yi]
                    labels[N,:] = t_labels[row,:]
                    labels[N, ei] = t_y
                    labels[N, ei+1] = 0
                    len_labels[N] = t_len_labs[row] + 1
                    relation_tables[N] = t_relation_tables[row]
                    relation_tables[N, ei] = predict_relation[row]
                    # attention_list[N] = t_alpha_list[row]
                    t_end += 1
                else:
                    #print(t_y)
                    ni = yi - t_end
                    labels[ni,:] = t_labels[row,:]
                    labels[ni, ei] = t_y
                    #print(ni, ei, labels[:,ei])
                    # attention_list[ni] = t_alpha_list[row].copy()
                    # attention_list[ni].append(alpha_np[row])
                    len_labels[ni] = t_len_labs[row] + 1 
                    logs[ni] = t_logs[yi]
                    relation_tables[ni] = t_relation_tables[row]
                    relation_tables[ni, ei] = predict_relation[row]
                    relation_stack[ni] = tt_relation_stack
                    labels_stack[ni] = tt_label_stack

                    #继续跟新下个输入。
                    res[ni], ys[ni] = t_p_re, t_p
                    hts[ni, :] = ht[row,:]
                    atts[ni, :] = attention_past[row, :]

                    #print(labels.cuda().numpy())
            # print(labels.cuda().numpy())
            # print(torch.exp(logs).cuda().numpy())
            # print(len_labels.cuda().numpy())
            # print(end_N)
            # N = B - end_N
            if N < 1:
                break

            #如果还没有结束，更新下个循环需要输入的变量。
            y_input = ys[:N]
            re_input = res[:N]
            ht_input = hts[:N, :]
            attention_past = atts[:N, :]
            ctx_input = ctx.repeat(N, 1, 1, 1)
            ctx_input_pool = ctx_pool.repeat(N, 1, 1, 1)
            ctx_key_object_input = ctx_key_object.repeat(N, 1, 1, 1)
            ctx_key_relation_input = ctx_key_relation.repeat(N, 1, 1, 1)
            ctx_mask_input = ctx_mask.repeat(N, 1, 1)
            ctx_mask_input_pool = ctx_mask_pool.repeat(N, 1, 1)
            p_mask_input = p_mask.repeat(N)


        logs = logs / len_labels
        _, index = torch.max(logs.unsqueeze(0),1)
        
        predict_object = labels[index[0],:].cuda().numpy()
        predict_relation = relation_tables[index[0]].cuda().numpy()
        len_predict = int(len_labels[index[0]].cuda().item())
        # print(labels.cuda().numpy())
        # print(logs.cuda().numpy())
        # print(len_labels.cuda().numpy())
        return predict_object[:len_predict+1], predict_relation[:len_predict+1]

    def find_parent(self, relation_stack, parent_stack):

        find_flag = 0
        p_re = 8
        p_y = parent_stack[-1]
        while relation_stack != []:
            if relation_stack[-1][:8].sum() > 0:
                for iii in range(9):
                    if relation_stack[-1][iii] != 0:
                        p_re = iii
                        p_y = parent_stack[-1]
                        relation_stack[-1][iii] = 0
                        if relation_stack[-1][:8].sum() == 0:
                            relation_stack.pop()
                            parent_stack.pop()
                        find_flag = 1
                        break
            else:
                relation_stack.pop()
                parent_stack.pop()

            if find_flag:
                break
        
        return p_re, p_y





