import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import DenseNet
from .decoder import Decoder
import math
import copy


class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.encoder = DenseNet(growthRate=params['growthRate'], 
                                reduction=params['reduction'],
                                bottleneck=params['bottleneck'], 
                                use_dropout=params['use_dropout'])
        self.init_context = nn.Linear(params['D'], params['n'])
        self.decoder = Decoder(params)
        self.object_criterion = nn.CrossEntropyLoss(reduction='none')
        self.relation_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.object_pix_criterion = nn.NLLLoss(reduction='none')


    def forward(self, params, x, x_mask, C_y,  
                P_y, C_re, P_re, y_mask, re_mask, length):

        #encoder
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        #decoder
        predict_objects, predict_relations, predict_objects_pix = self.decoder(ctx, ctx_mask, 
            C_y, P_y, y_mask, P_re, init_state, length)

        # compute loss 
        predict_objects = predict_objects.view(-1, predict_objects.shape[2])
        object_loss = self.object_criterion(predict_objects, C_y.view(-1))
        object_loss = object_loss.view(C_y.shape[0], C_y.shape[1])
        object_loss = ((object_loss * y_mask).sum(0) / y_mask.sum(0)).mean()

        predict_objects_pix = predict_objects_pix.view(-1, predict_objects_pix.shape[2])
        object_pix_loss = self.object_pix_criterion(predict_objects_pix, C_y.view(-1))
        object_pix_loss = object_pix_loss.view(C_y.shape[0], C_y.shape[1])
        object_pix_loss = ((object_pix_loss * y_mask).sum(0) / y_mask.sum(0)).mean()
        

        relation_loss = predict_relations.view(-1, predict_relations.shape[2])
        relation_loss = self.relation_criterion(relation_loss, C_re.view(-1, C_re.shape[2]))
        relation_loss = relation_loss.view(C_re.shape[0], C_re.shape[1], C_re.shape[2])
        relation_loss = (relation_loss * re_mask[:, :, None]).sum(2).sum(0) / re_mask.sum(0)
        relation_loss = relation_loss.mean()

        

        loss = params['lc_lambda'] * object_loss + \
               params['lr_lambda'] * relation_loss + \
               params['lc_lambda_pix'] * object_pix_loss

        return loss, object_loss, relation_loss

    def greedy_inference(self, x, x_mask, max_length, p_y, p_re, p_mask):
        
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        B, H, W = ctx_mask.shape
        attention_past = torch.zeros(B, 1, H, W).cuda()

        ctx_key_object = self.decoder.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation = self.decoder.conv_key_relation(ctx).permute(0, 2, 3, 1)
        
        relation_table = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static = torch.zeros(B, max_length, 9)
        P_masks = torch.zeros(B, max_length).cuda()
        predict_childs = torch.zeros(max_length, B).to(torch.long).cuda()

        ht = init_state
        ht_relation = init_state
        for i in range(max_length):
            predict_child, ht, attention, ct = self.decoder.get_child(ctx, ctx_key_object, ctx_mask, 
                attention_past, p_y, p_mask, p_re, ht)
            
            predict_childs[i]  = torch.argmax(predict_child, dim=1)
            predict_childs[i] *= p_mask.to(torch.long)
            attention_past = attention[:, None, :, :] + attention_past

            predict_relation, ht_relation = self.decoder.get_relation(ctx, ctx_key_relation, ctx_mask,
                predict_childs[i], ht_relation, ct) #(B, 9)

            P_masks[:, i] = p_mask 

            predict_relation_static[:, i, :] = predict_relation
            relation_table[:, i, :] = (predict_relation > 0)
            relation_table_static[:, i, :] = (predict_relation > 0)

            relation_table[:, :, 8] = relation_table[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table[ii, ji, iii] != 0:
                                p_re[ii] = iii
                                p_y[ii] = predict_childs[ji, ii]
                                relation_table[ii, ji, iii] = 0
                                relation_table[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        return predict_childs, P_masks, relation_table_static, predict_relation_static
    
    def greedy_inference_character_only(self, x, x_mask, max_length, p_y, p_re, s_re, p_mask):
        
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        init_state = torch.tanh(self.init_context(ctx_mean))  # (batch,n)

        B, H, W = ctx_mask.shape
        attention_past = torch.zeros(B, 1, H, W).cuda()

        ctx_key_object = self.decoder.conv_key_object(ctx).permute(0, 2, 3, 1)
        # ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        # ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation = self.decoder.conv_key_relation(ctx).permute(0, 2, 3, 1)
        
        relation_table = torch.zeros(B, max_length, 9).to(torch.long)
        relation_table_static = torch.zeros(B, max_length, 9).to(torch.long)
        predict_relation_static = torch.zeros(B, max_length, 9)
        P_masks = torch.zeros(B, max_length).cuda()
        predict_childs = torch.zeros(max_length, B).to(torch.long).cuda()

        ht = init_state
        ht_relation = init_state
        for i in range(max_length):
            predict_child, ht, attention, ct = self.decoder.get_child(ctx, ctx_key_object, ctx_mask, 
                attention_past, p_y, p_mask, p_re, ht)
            
            predict_childs[i]  = torch.argmax(predict_child, dim=1)
            predict_childs[i] *= p_mask.to(torch.long)
            attention_past = attention[:, None, :, :] + attention_past

            # predict_relation, ht_relation = self.decoder.get_relation(ctx, ctx_key_relation, ctx_mask,
            #     predict_childs[i], ht_relation, ct) #(B, 9)

            P_masks[:, i] = p_mask 

            # predict_relation_static[:, i, :] = predict_relation
            # relation_table[:, i, :] = (predict_relation > 0)
            relation_table[:, i, :] = s_re[:, i, :]

            # relation_table[:, :, 8] = relation_table[:, :, :8].sum(2)

            #print(i, predict_childs[i], predict_relation)
            #print(relation_table[:, :, 8])
            #greedy catch
            find_number = 0
            for ii in range(B):
                if p_mask[ii] < 0.5:
                    continue
                ji = i
                find_flag = 0
                while ji >= 0 :
                    if relation_table[ii, ji, 8]>0:
                        for iii in range(9):
                            if relation_table[ii, ji, iii] != 0:
                                p_re[ii] = iii
                                p_y[ii] = predict_childs[ji, ii]
                                relation_table[ii, ji, iii] = 0
                                relation_table[ii, ji, 8] -= 1
                                find_flag = 1
                                break
                        if find_flag:
                            break
                    ji -= 1
                find_number += find_flag
                if not find_flag:
                    p_mask[ii] = 0.
            # print(predict_childs[i], find_number, p_re, p_y, p_mask[ii])
            # print(relation_table_static[:, i, :])
            if find_number == 0:
                break
        return predict_childs, P_masks, relation_table_static, predict_relation_static

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






