import time
import os
import re
import numpy as np 
import random
import pickle as pkl
import torch
from torch.cuda.amp import autocast as autocast
from torch import optim, nn
from utils.utils import setup_seed,load_dict, prepare_bidirection_data, gen_sample, weight_init, compute_wer, compute_sacc, cmp_result,prepare_data
from utils.utils import update_lr as update_lr
from model.encoder_decoder_asyn_hard_attention_focal_ct_concat import Encoder_Decoder_Bi_Asyn as Encoder_Decoder
from utils.data_iterator import dataIterator, BatchBucket
from utils.gtd import gtd2latex, relation2tree, latex2gtd
import copy
from logger import Logger
import sys
from utils.gtd import reverse_ver_3, re_id
from load_encoder import *

EXP_ID = 1
setup_seed(20230406)
pretrain_path = 'pretrain_offline.pkl'
os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
# EXP_ID = sys.argv[1]
bfs2_path = '../CROHME/'
work_path = './'
# save_path = '../../../../../nmt/chb_bitd/'
# EXPpath = 'result_TDv2_Bi-TD-asyn-hardattn-add-gru/' + str(EXP_ID) + '/'
EXPpath = 'result_asyn-TD-final_test/' + str(EXP_ID) + '/'
folder = EXPpath
if not os.path.exists(EXPpath):  #判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(EXPpath)
# if not os.path.exists(save_path):  #判断是否存在文件夹如果不存在则创建为文件夹
#     os.makedirs(save_path)
sys.stdout = Logger(work_path + EXPpath + 'Bi_TDv2.txt', sys.stdout)
sys.stderr = Logger(work_path + EXPpath + 'Bi_TDv2.txt', sys.stderr)

# whether use multi-GPUs
multi_gpu_flag = False
# whether init params
init_param_flag = True
# whether reload params
reload_flag = False
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dictionaries = [bfs2_path + 'dictionary_107.txt', bfs2_path + 'dictionary_relation_9.txt']
bidirection_datasets = [bfs2_path + 'train_images.pkl', bfs2_path + 'train_chb_bidirection_label_gtd.pkl', bfs2_path + 'train_chb_bidirection_relations.pkl']
dataset = [bfs2_path + 'train_images.pkl', bfs2_path + 'train_chb_label_gtd.pkl', bfs2_path + 'train_chb_relations.pkl']
valid_datasets = [bfs2_path + '14_test_images.pkl', bfs2_path + '14_chb_test_label_gtd.pkl', bfs2_path + '14_chb_test_relations.pkl']
valid_output = [work_path+ EXPpath +'/symbol_relation/', work_path+ EXPpath +'/memory_alpha/']
valid_result = [work_path+ EXPpath +'/valid.cer', work_path+ EXPpath +'/valid.exprate']
saveto_wer = work_path+ EXPpath + '/WAP_params_wer'
saveto_latex = work_path+ EXPpath + '/WAP_params_exprate'
last_saveto = work_path + EXPpath + '/WAP_params_last.pkl'



# training settings
maxlen = 200
max_epochs = 201
lrate = 2
my_eps = 1e-6
decay_c = 1e-4
clip_c = 100.
DECAY_TIMES = 2
DECAY_RATE = 10
# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 15
validStart = 0
finish_after = 10000000

# model architecture
params = {}
params['n'] = 256
params['m'] = 256
params['re_m'] = 64
params['dim_attention'] = 512
params['D'] = 936
params['K'] = 107
params['Kre'] = 9
params['mre'] = 256
params['maxlen'] = maxlen

params['growthRate'] = 24
params['reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 1

params['lc_lambda'] = 1.
params['lr_lambda'] = 1.
params['lc_lambda_pix'] = 0.5

params['L2R-R2L'] = 1
params["KL"] = False
params['DIRECTION'] = 'L2R'

# load dictionary
worddicts = load_dict(dictionaries[0])
print ('total chars',len(worddicts))
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk
reworddicts = load_dict(dictionaries[1])
print ('total relations',len(reworddicts))
reworddicts_r = [None] * len(reworddicts)
for kk, vv in reworddicts.items():
    reworddicts_r[vv] = kk
# load valid gtd
with open(valid_datasets[1], 'rb') as fp:
    valid_gtds = pkl.load(fp)

# train_dataIterator = BatchBucket(600, 2100, 200, 800000, 1,  #batch size 
#                     datasets[0], datasets[1], datasets[2], 
#                     dictionaries[0], dictionaries[1])
# train, train_uid = train_dataIterator.get_batches()

train_dataIterator = BatchBucket(600, 2100, 200, 800000*3, 30,  #batch size 
                    bidirection_datasets[0], bidirection_datasets[1], bidirection_datasets[2], 
                    dictionaries[0], dictionaries[1], direction="L2R-R2L")
train, train_uid = train_dataIterator.get_bidirection_batches()

valid, valid_uid = dataIterator(valid_datasets[0], valid_datasets[1], 
                    valid_datasets[2], worddicts, reworddicts,
                    8, 8000000, 200, 8000000)  


scaler = torch.cuda.amp.GradScaler()
# display
uidx = 0  # count batch
lpred_loss_s = 0.  # count loss
repred_loss_s = 0.
loss_s = 0.
KL_loss_s = 0.
attn_loss_s = 0.
l2r_s = 0.
r2l_s = 0.
ud_s = 0  # time for training an epoch
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100

WER = 100
LATEX_ACC = 0


# initialize model
WAP_model = Encoder_Decoder(params)

# freezze_params(WAP_model, pretrain_path)
# for p in WAP_model.parameters():
#     print(p.requires_grad)
if init_param_flag:
    WAP_model.apply(weight_init)
    load_partial_params(WAP_model, pretrain_path)
if multi_gpu_flag:
    WAP_model = nn.DataParallel(WAP_model, device_ids=[0, 1, 2, 3])
if reload_flag:
    WAP_model.load_state_dict(torch.load(last_saveto,map_location=lambda storage,loc:storage))
    

WAP_model.cuda()

# print model's parameters
# model_params = WAP_model.named_parameters()
# for k, v in model_params:
#     print(k)

# loss function
# criterion = torch.nn.CrossEntropyLoss(reduce=False)
# optimizer
optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, WAP_model.parameters()), lr=lrate, eps=my_eps, weight_decay=decay_c)


print('Optimization')

# statistics
history_errs = []

for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()

    train_dataIterator._reset()
    train, train_uid = train_dataIterator.get_bidirection_batches()
    random.shuffle(train)
    if validFreq == -1:
        validFreq = len(train)
    if saveFreq == -1:
        saveFreq = len(train)
    if sampleFreq == -1:
        sampleFreq = len(train)
    for x, ly, ry, re, ma, lp, rp in train:
        with autocast():
            WAP_model.train()
            ud_start = time.time()
            n_samples += len(x)
            uidx += 1
            x, x_mask, ly, y_mask, ry, re, ma, ma_mask, lp, rp  = \
                    prepare_bidirection_data(params, x, ly, ry, re, ma, lp, rp)

            # lp,rp: [lp_l2r, lp_r2l] : len,batch
            # x : batch, c, w, h
            # x_mask: batch, w, h
            # ly : [ly_l2r, ly_r2l] : len,batch
            # re : [re_l2r, re_r2l] : len,batch
            # ma : [ma_l2r, ma_r2l] : len, batch 9

            # l: left = child
            # r: right = parent
            x = torch.from_numpy(x).cuda()
            x_mask = torch.from_numpy(x_mask).cuda()

            ly_L2R, ly_R2L = ly
            y_mask_L2R, y_mask_R2L= y_mask
            ry_L2R, ry_R2L = ry
            re_L2R, re_R2L = re
            ma_L2R, ma_R2L = ma
            ma_mask_L2R, ma_mask_R2L = ma_mask
            lp_L2R, lp_R2L = lp
            rp_L2R, rp_R2L = rp

            length = ly_L2R.shape[0]  # L2R/L2R的方向必定一致

        

            # tensor for L2R
            ly_L2R = torch.from_numpy(ly_L2R).to(torch.long).cuda()  # (seqs_y,batch)
            y_mask_L2R = torch.from_numpy(y_mask_L2R).cuda()  # (seqs_y,batch)
            ry_L2R = torch.from_numpy(ry_L2R).to(torch.long).cuda()  # (seqs_y,batch)
            re_L2R = torch.from_numpy(re_L2R).to(torch.long).cuda()  # (seqs_y,batch)
            ma_L2R = torch.from_numpy(ma_L2R).cuda()  
            ma_mask_L2R = torch.from_numpy(ma_mask_L2R).cuda()
            lp_L2R = torch.from_numpy(lp_L2R).to(torch.long).cuda()
            rp_L2R = torch.from_numpy(rp_L2R).to(torch.long).cuda()

            # tensor for R2L
            ly_R2L = torch.from_numpy(ly_R2L).to(torch.long).cuda()  # (seqs_y,batch)
            y_mask_R2L = torch.from_numpy(y_mask_R2L).cuda()  # (seqs_y,batch)
            ry_R2L = torch.from_numpy(ry_R2L).to(torch.long).cuda()  # (seqs_y,batch)
            re_R2L = torch.from_numpy(re_R2L).to(torch.long).cuda()  # (seqs_y,batch)
            ma_R2L = torch.from_numpy(ma_R2L).cuda()  
            ma_mask_R2L = torch.from_numpy(ma_mask_R2L).cuda()
            lp_R2L = torch.from_numpy(lp_R2L).to(torch.long).cuda()
            rp_R2L = torch.from_numpy(rp_R2L).to(torch.long).cuda()

            ly = [ly_L2R, ly_R2L ]
            y_mask = [y_mask_L2R, y_mask_R2L ]
            ry = [ry_L2R, ry_R2L ] 
            re = [re_L2R, re_R2L ]
            ma = [ma_L2R, ma_R2L ] 
            ma_mask = [ma_mask_L2R, ma_mask_R2L ]
            lp = [lp_L2R, lp_R2L ]
            rp = [rp_L2R, rp_R2L ]

            loss, object_loss, relation_loss =  WAP_model(params, x, x_mask, 
                ly, ry, ma, re, y_mask, ma_mask, lp, rp, length)
            lpred_loss_s += object_loss.item()
            repred_loss_s += relation_loss.item()
            # attn_loss_s += attn_loss.item()
            loss_s += loss.item()
            # KL_loss_s += KL_loss.item()




        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        #loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(WAP_model.parameters(), clip_c)

        # update
        # scaler.scale(loss).backward()
        new_lr = update_lr(optimizer, eidx, uidx, len(train), max_epochs, lrate)

        scaler.step(optimizer)
        scaler.update()

        

        ud = time.time() - ud_start
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            lpred_loss_s /= dispFreq
            repred_loss_s /= dispFreq
            KL_loss_s /= dispFreq
            attn_loss_s /= dispFreq
            # l2r_s /=dispFreq
            # r2l_s /= dispFreq
            print ('Epoch', eidx, ' Update', uidx, ' Cost_object %.7f, Cost_relation %.7f, attn_loss: %.7f' % \
                (np.float64(lpred_loss_s),  np.float64(repred_loss_s) , np.float64(attn_loss_s)), \
                ' UD %.3f' % ud_s, ' lrate', new_lr, ' eps', my_eps, ' bad_counter', bad_counter)
            # print("l2r: ",l2r_s,"r2l: ", r2l_s)
            ud_s = 0
            loss_s = 0.
            lpred_loss_s = 0.
            repred_loss_s = 0.
            KL_loss_s = 0.
            attn_loss_s = 0.

        if np.mod(uidx, saveFreq) == 0:
            print('Saving latest model params ... ')
            torch.save(WAP_model.state_dict(), last_saveto)
        
        # validation
        if np.mod(uidx, sampleFreq) == 0 and (eidx % 2) == 0:
        # if True:
            number_right = 0
            total_distance = 0
            total_length = 0
            latex_right = 0
            total_latex_distance = 0
            total_latex_length = 0
            total_number = 0

            print('begin sampling')
            ud_epoch_train = (time.time() - ud_epoch) / 60.
            print('epoch training cost time ... ', ud_epoch_train)
            WAP_model.eval()
            
            fp_results = open(work_path + EXPpath + 'reuslts-debug_focal.txt', 'w')
            with autocast():
                with torch.no_grad():
                    valid_count_idx = 0
                    for x, ly, ry, re, ma, lp, rp in valid:
                        x, x_mask, C_y, y_mask, P_y, P_re, C_re, C_re_mask, lp, rp = \
                        prepare_data(params, x, ly, ry, re, ma, lp, rp)
                
                        L, B = C_y.shape[:2]
                        x = torch.from_numpy(x).cuda()  # (batch,1,H,W)
                        x_mask = torch.from_numpy(x_mask).cuda()  # (batch,H,W)

                        # ly, _ = ly
                        # y_mask, _= y_mask
                        # ry, _ = ry
                        # re, _ = re
                        # ma, _ = ma
                        # ma_mask, _ = ma_mask
                        # lp, _ = lp
                        # rp, _ = rp

                        lengths_gt = (y_mask > 0.5).sum(0)
                        y_mask = torch.from_numpy(y_mask).cuda()  # (seqs_y,batch)
                        P_y = torch.from_numpy(P_y).to(torch.long).cuda()  # (seqs_y,batch)
                        P_re = torch.from_numpy(P_re).to(torch.long).cuda()  # (seqs_y,batch)

                        object_predicts, P_masks, relation_table_static, _ \
                            = WAP_model.greedy_inference(x, x_mask, L+1, copy.deepcopy(P_y[0]), copy.deepcopy(P_re[0]), copy.deepcopy(y_mask[0]))

                        object_predicts, P_masks = object_predicts.cpu().numpy(), P_masks.cpu().numpy()   # 原因有可能是数据处理阶段position和C_y
                        relation_table_static = relation_table_static.numpy()
    

                        for bi in range(B):
                            length_predict = min((P_masks[bi, :] > 0.5).sum() + 1, P_masks.shape[1])
                            object_predict = object_predicts[:int(length_predict), bi]
                            relation_predict = relation_table_static[bi, :int(length_predict), :]
                            gtd = relation2tree(object_predict[1::2], relation_predict[1::2], worddicts_r, reworddicts_r)
                            latex = gtd2latex(gtd)
                            uid = valid_uid[total_number]
                            groud_truth_gtd = valid_gtds[uid]       
                            if(params['DIRECTION']  == 'R2L'):      
                                groud_truth_gtd = reverse_ver_3(groud_truth_gtd, reworddicts)
                                groud_truth_gtd = re_id(groud_truth_gtd)
                            groud_truth_latex = gtd2latex(groud_truth_gtd)
                            
                            child = C_y[:int(lengths_gt[bi]), bi]
                            distance, length = cmp_result(object_predict[:-1], child)
                            total_number += 1
                            

                            if distance == 0:
                                number_right += 1
                                fp_results.write(uid + 'Object True\t')
                            else:
                                fp_results.write(uid + 'Object False\t')
                            
                            latex_distance, latex_length = cmp_result(groud_truth_latex, latex)
                            if latex_distance == 0:
                                latex_right += 1
                                fp_results.write('Latex True\n')
                            else:
                                fp_results.write('Latex False\n')


                            total_distance += distance
                            total_length += length
                            total_latex_distance += latex_distance
                            total_latex_length += latex_length

                            fp_results.write(groud_truth_latex+'\n')
                            fp_results.write(latex+'\n')
                            
                            for li in range(lengths_gt[bi]):
                                fp_results.write(worddicts_r[child[li]] + ' ')
                            fp_results.write('\n')

                            for li in range(length_predict):
                                fp_results.write(worddicts_r[object_predict[li]] + ' ')
                            fp_results.write('\n')

                
            wer = total_distance / total_length * 100
            sacc = number_right / total_number * 100
            latex_wer = total_latex_distance / total_latex_length * 100
            latex_acc = latex_right / total_number * 100
            fp_results.close()
                        
            print('valid set decode done')
            ud_epoch = (time.time() - ud_epoch) / 60.
            print('WER', wer, 'SACC', sacc, 'Latex WER', latex_wer, 'Latex SACC', latex_acc,  'epoch cost time ... ', ud_epoch)
            # the first time validation or better model
            if latex_wer <= WER :
                WER = latex_wer
                if(latex_wer < 10):
                    print('Saving best model params ... ')
                    torch.save(WAP_model.state_dict(), saveto_wer + '_' + str(WER)+'.pkl')
            if latex_acc >= LATEX_ACC:
                LATEX_ACC = latex_acc
                if latex_acc>56.5:
                    print('Saving best model params ... ')
                    torch.save(WAP_model.state_dict(), saveto_latex + '_' + str(LATEX_ACC)+'.pkl')
            elif(latex_acc >57):
                torch.save(WAP_model.state_dict(), saveto_latex + '_extra' +'_' + str(latex_acc)+'.pkl')
            print("min wer: ", WER, "max SACC: ",LATEX_ACC)




        # finish after these many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            estop = True
            break

    print('Seen %d samples' % n_samples)

    # early stop
    if estop:
        break
        

        

    