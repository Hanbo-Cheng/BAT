import torch
# from model.encoder_decoder_only_optical import Encoder_Decoder


def load_partial_params(model, path):
    pretrain_encoder_path = path
    pretrained_dict = torch.load(pretrain_encoder_path)
    model_dict = model.state_dict()
    for k,v in pretrained_dict.items():
        if(k in model_dict.keys()):
            model_dict[k] = v
        #     print('load: ',k)
        # else:
        #     print('fail: ',k)
    model.load_state_dict(model_dict)


def load_pretrained_online_encoder(model, path):
    pretrain_encoder_path = path
    pretrained_dict = torch.load(pretrain_encoder_path)
    model_dict = model.state_dict()
    # for k,k2 in zip(model_dict.items(), pretrained_dict.items()):
    #     if("encoder" in  k[0]):
    #         print(k[0], k[1].shape)
    #         print(k2[0],k2[1].shape)
    # for k,v in pretrained_dict.items():
    #         if("encoder" in  k):
    #             print(k,v.shape)

    for k,v in pretrained_dict.items():
        if("encoder" in  k):
            list = k.split('.')
            list[0] = "encoder_online"
            k = '.'.join(list)
            if(len(model_dict[k].shape)>2 and model_dict[k].shape[1] != v.shape[1]):
                # print(v.shape)
                # print(model_dict[k].shape)
                v = v.repeat(1, int(model_dict[k].shape[1]/v.shape[1]),1 ,1)
                # print(v.shape)
            model_dict[k] = v
    model.load_state_dict(model_dict)
def load_pretrained_offline_encoder(model, path):
    pretrain_encoder_path = path
    pretrained_dict = torch.load(pretrain_encoder_path)
    model_dict = model.state_dict()
    # for k,k2 in zip(model_dict.items(), pretrained_dict.items()):
    #     if("encoder" in  k[0]):
    #         print(k[0], k[1].shape)
    #         print(k2[0],k2[1].shape)
    # for k,v in pretrained_dict.items():
    #         if("encoder" in  k):
    #             print(k,v.shape)

    for k,v in pretrained_dict.items():
        if("encoder" in  k):
            list = k.split('.')
            list[0] = "encoder_offline"
            k = '.'.join(list)
            print(k)
            # if(len(model_dict[k].shape)>2 and model_dict[k].shape[1] != v.shape[1]):
            #     # print(v.shape)
            #     # print(model_dict[k].shape)
            #     v = v.repeat(1, int(model_dict[k].shape[1]/v.shape[1]),1 ,1)
            #     # print(v.shape)
            model_dict[k] = v
    model.load_state_dict(model_dict)

def load_pretrained_encoder(model, path):
    pretrain_encoder_path = path
    pretrained_dict = torch.load(pretrain_encoder_path)
    model_dict = model.state_dict()

    for k,v in pretrained_dict.items():
        if("encoder" in  k):
            model_dict[k] = v
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    maxlen = 200
    max_epochs = 5000
    lrate = 1
    my_eps = 1e-6
    decay_c = 1e-4
    clip_c = 100.
    DECAY_TIMES = 5
    DECAY_RATE = 2
    # early stop
    estop = False
    halfLrFlag = 0
    bad_counter = 0
    patience = 15
    validStart = 0
    finish_after = 10000000
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

    params['time_idx'] = True
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    if(params['time_idx']):
        params['input_channels'] = 4
    else:
        params['input_channels'] = 3

    params['lc_lambda'] = 1.
    params['lr_lambda'] = 1.
    params['lc_lambda_pix'] = 0.5

    model = Encoder_Decoder(params)

    pretrain_encoder_path = "WAP_params_exprate_54.361054766734284.pkl"
    load_pretrained_online_encoder(model=model ,path=pretrain_encoder_path)

    
