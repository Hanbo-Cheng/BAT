import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder_L2R(nn.Module):
    '''
    主解码器，从左到右
    '''
    def __init__(self, params):
        super(Decoder_L2R, self).__init__()
        self.object_embedding = nn.Embedding(params['K'], params['m'])
        self.relation_embedding = nn.Embedding(params['Kre'], params['re_m'])

        self.gru00 = nn.GRUCell(params['m']+params['re_m'], params['n'])
        self.gru01 = nn.GRUCell(params['D'] , params['n']) # 修改了context 的size，用于处理context与hidden state
        self.gru10 = nn.GRUCell(params['D'], params['n'])
        self.gru11 = nn.GRUCell(params['D'], params['n'])  #同理

        self.conv_key_object = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.object_attention = CoverageAttention(params)
        self.object_probility = ObjectProbility(params)
        self.object_hidden_attention = Hidden_state_Attention(params)
        
        self.fc_key_R2L_hidden_ch = nn.Linear(params['n'], params['dim_attention'])  # 两个FC，用于处理hidden state 得到key
        self.fc_key_R2L_hidden_re = nn.Linear(params['n'], params['dim_attention'])
        self.conv_key_relation = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.relation_attention = Attention(params)
        self.relation_probility = RelationProbility(params)
        self.relation_hidden_attention = Hidden_state_Attention(params)

        self.fc_ct = nn.Linear(params['D'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_probility = nn.Linear(int(params['m'] / 2), params['K'])


        self.param_K = params['K']
        self.param_Kre = params['Kre']

    def forward(self, ctx_val, ctx_mask, C_ys, P_ys, P_y_masks, 
                P_res, init_state, length, R2L_hidden_ch, R2L_hidden_re, R2L_hidden_mask):
        #print(P_ys)
        ht = init_state
        ht_lm = init_state
        ht_relation = init_state
        B, H, W = ctx_mask.shape
        r2l_maxlen = R2L_hidden_ch.shape[0]
        #attentions = torch.zeros(length, B, H, W).cuda()
        attention_past = torch.zeros(B, 1, H, W).cuda()
        hidden_attention_past_ch = torch.zeros(B, 1, r2l_maxlen).cuda()
        hidden_attention_past_re = torch.zeros(B, 1, r2l_maxlen).cuda()
        hidden_attention_ch = torch.zeros(length, B, r2l_maxlen).cuda()
        predict_childs = torch.zeros(length, B, self.param_K).cuda()
        predict_childs_lm = torch.zeros(length, B, self.param_K).cuda()
        predict_childs_pix = torch.zeros(length, B, self.param_K).cuda()
        predict_relations = torch.zeros(length, B, self.param_Kre).cuda()

        ctx_key_object = self.conv_key_object(ctx_val).permute(0, 2, 3, 1)
        ctx_key_relation = self.conv_key_relation(ctx_val).permute(0, 2, 3, 1)

        R2L_hidden_key_ch = self.fc_key_R2L_hidden_ch(R2L_hidden_ch) # 用于得到obj与relation的hidden state的key
        R2L_hidden_key_re = self.fc_key_R2L_hidden_re(R2L_hidden_re)


        predict_features = self.get_predicts(ctx_val) #(Batch, H, W, k)
        for i in range(length):
            predict_childs[i], predict_childs_lm[i], ht, ht_lm, attention, attention_hidden, ct = self.get_child(ctx_val, ctx_key_object, ctx_mask, 
                attention_past, hidden_attention_past_ch, P_ys[i], P_y_masks[i], P_res[i], ht, ht_lm, R2L_hidden_ch, R2L_hidden_key_ch, R2L_hidden_mask) # 增加了输入R2L的V与K
            attention_past = attention[:, None, :, :] + attention_past
            hidden_attention_past_ch = attention_hidden[:, None, :] + hidden_attention_past_ch
            hidden_attention_ch[i] = attention_hidden
            ##(Batch, H, W, k) --> (Batch, k)
            predict_childs_pix[i] = torch.log((attention.detach()[:, :, :, None] * predict_features).sum(2).sum(1))
            predict_relations[i], ht_relation, hidden_attention_re = self.get_relation(ctx_val, ctx_key_relation, 
                ctx_mask, C_ys[i], ht_relation, ct, R2L_hidden_re, R2L_hidden_key_re, R2L_hidden_mask, hidden_attention_past_re) # 增加了输入R2L的V与K
            hidden_attention_past_re = hidden_attention_re[:, None, :] + hidden_attention_past_re   
        return predict_childs, predict_childs_lm, predict_relations, predict_childs_pix, hidden_attention_ch

    def get_predicts(self, ctx_val):
        predict_features = self.fc_ct(ctx_val.permute(0, 2, 3, 1)) #(Batch, H, W, D) --> (Batch, H, W, m)
        # maxout
        predict_features = predict_features.view(predict_features.shape[0], predict_features.shape[1], 
                                    predict_features.shape[2], -1, 2)  #(Batch, H, W, d/2, 2)
        predict_features = predict_features.max(4)[0]  # (Batch, H, W, d/2)
        predict_features = self.dropout(predict_features)
        predict_features = self.fc_probility(predict_features)  # (Batch, H, W, k)
        predict_features = F.softmax(predict_features, dim=3) #(Batch, H, W, k)
        return predict_features

    def get_child(self, ctx_val, ctx_key, ctx_mask, attention_past, hidden_attention_past, p_y, p_y_mask, p_re, ht, ht_lm, R2L_hidden_ch, R2L_hidden_key_ch, R2L_hidden_mask):
        # print(p_y)
        # print(p_y.dtype)
        p_y = self.object_embedding(p_y)
        p_re = self.relation_embedding(p_re)
        p = torch.cat([p_y, p_re], dim=1)
        ht_hat = self.gru00(p, ht)  #att_query
        ht_hat = p_y_mask[:, None] * ht_hat + (1. - p_y_mask)[:, None] * ht

        ct_hidden, attention_hidden = self.object_hidden_attention(R2L_hidden_ch, R2L_hidden_key_ch, R2L_hidden_mask, hidden_attention_past,ht_hat_lm)
        ct, attention = self.object_attention(ctx_val, ctx_key, ctx_mask, attention_past, ht_hat)

        # for language modeling 
        ht_hat_lm = self.gru00(p, ht_lm)
        ht_hat_lm = p_y_mask[:, None] * ht_hat_lm + (1. - p_y_mask)[:, None] * ht_lm

        # for language modeling 
        ct_zero = torch.zeros_like(ct)
        ht_lm = self.gru01(ct_zero, ht_hat_lm)
        ht_lm = p_y_mask[:, None] * ht_lm + (1. - p_y_mask)[:, None] * ht_hat_lm

        ht = self.gru01(ct, ht_hat)
        ht = p_y_mask[:, None] * ht + (1. - p_y_mask)[:, None] * ht_hat

        predict_child = self.object_probility(ct, ht, p_y, p_re, ct_hidden)
        predict_child_lm = self.object_probility(ct_zero, ht_lm, p_y, p_re, ct_hidden)

        return predict_child, predict_child_lm, predict_child_lm, ht, ht_lm, attention, attention_hidden.permute(1,0), ct
    
    def get_relation(self, ctx_val, ctx_key, ctx_mask, c_y, ht, ct, R2L_hidden_re, R2L_hidden_key_re, R2L_hidden_mask, hidden_attention_past_re):
        c_y = self.object_embedding(c_y)
        ht_query = self.gru10(ct, ht)
        ct_hidden, hidden_attention_re = self.relation_hidden_attention(R2L_hidden_re, R2L_hidden_key_re, R2L_hidden_mask, hidden_attention_past_re, ht_query)
        # ct_hidden = torch.zeros_like(ct_hidden).cuda()
        # ct_hidden = self.ct_hidden_re_fc_1(ct_hidden)
        # ct_hidden = self.ct_hidden_re_fc_2(ct_hidden)
        # ht_query = ct_hidden + ht_query
        ct, _ = self.relation_attention(ctx_val, ctx_key, ctx_mask, ht_query)
        
        # ct_new = torch.concat((ct,ct_hidden), dim= -1)
        ht = self.gru11(ct, ht_query)
        predict_relation = self.relation_probility(ct, c_y, ht, ct_hidden)
        return predict_relation, ht, hidden_attention_re.permute(1,0)

class CoverageAttention(nn.Module):
    def __init__(self, params):
        super(CoverageAttention, self).__init__()
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_att_past = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)
        self.fc_att_past = nn.Linear(512, params['dim_attention'])
        self.fc_attention = nn.Linear(params['dim_attention'], 1)
    
    def forward(self, ctx_val, ctx_key, ctx_mask, attention_past, ht_query):

        ht_query = self.fc_query(ht_query)

        attention_past = self.conv_att_past(attention_past).permute(0, 2, 3, 1)
        attention_past = self.fc_att_past(attention_past) #(batch, H, W, dim_att)

        attention_score = torch.tanh(ctx_key + ht_query[:, None, None, :] + attention_past)
        attention_score = self.fc_attention(attention_score).squeeze(3)
        
        attention_score = attention_score - attention_score.max()
        attention_score = torch.exp(attention_score) * ctx_mask
        attention_score = attention_score / (attention_score.sum(2).sum(1)[:, None, None] + 1e-10)

        ct = (ctx_val * attention_score[:, None, :, :]).sum(3).sum(2)

        return ct, attention_score

class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.fc_attention = nn.Linear(params['dim_attention'], 1)
    
    def forward(self, ctx_val, ctx_key, ctx_mask, ht_query):

        ht_query = self.fc_query(ht_query)

        attention_score = torch.tanh(ctx_key + ht_query[:, None, None, :])
        attention_score = self.fc_attention(attention_score).squeeze(3)
        
        attention_score = attention_score - attention_score.max()
        attention_score = torch.exp(attention_score) * ctx_mask
        attention_score = attention_score / (attention_score.sum(2).sum(1)[:, None, None] + 1e-10)

        ct = (ctx_val * attention_score[:, None, :, :]).sum(3).sum(2)

        return ct, attention_score

class Hidden_state_Attention(nn.Module):
    '''
    在 第一个维度上做注意力操作
    '''
    def __init__(self, params):
        super(Hidden_state_Attention, self).__init__()
        self.conv_att_past = nn.Conv1d(1, 512, kernel_size=3, bias=False, padding=1)
        self.fc_query = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.fc_att_past = nn.Linear(512, params['dim_attention'])
        self.fc_attention = nn.Linear(params['dim_attention'], 1)
    
    def forward(self, ctx_val, ctx_key, ctx_mask, attention_past ,ht_query):
        '''
        ctx_val: maxlen, b, 256 
        ctx_key: maxlen, b, 256
        ctx_mask maxlen b
        ht_query : b,256
        '''
        ht_query = self.fc_query(ht_query)
        attention_past = self.conv_att_past(attention_past).permute(2,0,1)
        attention_past = self.fc_att_past(attention_past)
        attention_score = torch.tanh(ctx_key + ht_query[None,:, :] + attention_past) 
        attention_score = self.fc_attention(attention_score).squeeze(-1) # attention_score: maxlen,b
        
        # attention_score = attention_score - attention_score.max()
        # attention_score = torch.exp(attention_score) * ctx_mask
        # attention_score = attention_score / (attention_score.sum(0)[None,:] + 1e-10)
        # # attention_score_hard = (attention_score >= 0.3).to(torch.float32)
        # ct = (ctx_val * (attention_score)[:, :, None]).sum(0)
        # # TODO 是否将hard attn作为past alpha  hard 传出 训练效果不佳

        attention_score_hard = (attention_score >= 0).to(torch.float32)
        attention_score_hard = attention_score_hard * ctx_mask
        attention_score_hard = attention_score_hard / (attention_score_hard.sum(0) + 1e-10)
        ct = (ctx_val * (attention_score_hard)[:, :, None]).sum(0)
        return ct, attention_score_hard  # best : use hardattn

# class Context_Hidden_state_Attention(nn.Module):
#     '''
#     在最后一个维度上做注意力操作
#     '''
#     def __init__(self, params):
#         super(Hidden_state_Attention, self).__init__()
#         self.fc_query = nn.Linear(params['n']*2, params['dim_attention'], bias=False)
#         self.fc_attention = nn.Linear(params['dim_attention'], params['n'])
    
#     def forward(self, ctx_val, ctx_key, ctx_mask, ht_query):
#         '''
#         ctx_val: b, 512 
#         ctx_key: b, 512
#         ctx_mask
#         ht_query : b,256
#         '''
#         ht_query = self.fc_query(ht_query)

#         attention_score = torch.tanh(ctx_key + ht_query[:, :]) 
#         attention_score = self.fc_attention(attention_score).squeeze(-1) # attention_score: maxlen,b
        
#         attention_score = attention_score - attention_score.max()
#         attention_score = torch.exp(attention_score) * ctx_mask
#         attention_score = attention_score / (attention_score.sum(0)[None,:] + 1e-10)

#         ct = (ctx_val * attention_score[:, :, None]).sum(0)

#         return ct, attention_score

class ObjectProbility(nn.Module):
    def __init__(self, params):
        super(ObjectProbility, self).__init__()
        self.fc_ct = nn.Linear(params['D'], params['m'])  # 为了适应新的输入增加了256个宽度
        self.fc_ht = nn.Linear(params['n'], params['m'])
        self.fc_ctht = nn.Linear(params['n'], params['m'])
        self.fc_p_y = nn.Linear(params['m'], params['m'])
        self.fc_p_re = nn.Linear(params['re_m'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_probility = nn.Linear(int(params['m'] / 2), params['K'])
        

    def forward(self, ct, ht, p_y, p_re, ct_h):
        clogit = self.fc_ct(ct) + self.fc_ht(ht) + self.fc_p_y(p_y) + self.fc_p_re(p_re) + self.fc_ctht(ct_h)
        # maxout 
        clogit = clogit.view(clogit.shape[0], -1, 2)  
        clogit = clogit.max(2)[0]  
        clogit = self.dropout(clogit)
        cprob = self.fc_probility(clogit)
        return cprob

class RelationProbility(nn.Module):
    def __init__(self, params):
        super(RelationProbility, self).__init__()
        self.fc_ct = nn.Linear(params['D'], params['mre'])
        self.fc_c_y = nn.Linear(params['m'], params['mre'])
        self.fc_ht = nn.Linear(params['n'], params['m'])
        self.fc_ctht = nn.Linear(params['n'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_probility = nn.Linear(params['mre']//2, params['Kre'])
        # self.fc_adaptive = nn.Linear(params['Kre'], params['Kre'])

    def forward(self, ct, c_y, ht, ct_h):
        re_probility = self.fc_ct(ct) + self.fc_c_y(c_y) + self.fc_ht(ht) + self.fc_ctht(ct_h)
        re_probility = re_probility.view(re_probility.shape[0], -1, 2)
        re_probility = re_probility.max(2)[0]
        re_probility = self.dropout(re_probility)
        re_probility = self.fc_probility(re_probility)
        # re_probility = re_probility  + self.fc_adaptive(re_probility)  # adding adaptive threshold
        return re_probility
