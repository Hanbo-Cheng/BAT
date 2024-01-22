import numpy as np
import os
import pdb
import random
import sys
import pickle as pkl
import copy
# from utils import load_dict

def relation2tree(childs, relations, worddicts_r, reworddicts_r):
    gtd = [[] for c in childs]
    start_relation = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    relation_stack = [start_relation]
    parent_stack = [(106, 0)]
    p_re = 0
    p_y = 106
    p_id = 0

    for ci, c in enumerate(childs):
        gtd[ci].append(worddicts_r[c])
        gtd[ci].append(ci+1)
        
        find_flag = 0
        while relation_stack != []:
            if relation_stack[-1][:8].sum() > 0:
                for iii in range(9):
                    if relation_stack[-1][iii] != 0:
                        p_re = iii
                        p_y, p_id = parent_stack[-1]
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
        
        if not find_flag:
            p_y = childs[ci-1]
            p_id = ci
            p_re = 8
        gtd[ci].append(worddicts_r[p_y])
        gtd[ci].append(p_id)
        gtd[ci].append(reworddicts_r[p_re])

        relation_stack.append(relations[ci])
        parent_stack.append((c, ci+1))

    return gtd


def gen_gtd_align(gtd):
    wordNum = len(gtd)
    align = np.zeros([wordNum, wordNum], dtype='int8')
    wordindex = -1

    for i in range(len(gtd)):
        wordindex += 1
        parts = gtd[i]
        if len(parts) == 5:
            realign = parts[3]
            # import pdb;pdb.set_trace()
            realign_index = int(str(realign))
            align[realign_index, wordindex] = 1
    return align

def gen_gtd_relation_align(gtd,dict):
    wordNum = len(gtd)
    align = np.zeros([wordNum, 9], dtype='int8')
    wordindex = -1
    new_gtd = re_id(copy.deepcopy(gtd))
    # for i,j in zip(gtd,new_gtd):
    #     if(i != j):
    #         print(gtd)
    #         break

    for i in range(len(new_gtd)):
        wordindex += 1
        parts = new_gtd[i]
        
        if len(parts) == 5:
            relation = dict[parts[-1]]
            realign = parts[3]
            # import pdb;pdb.set_trace()
            realign_index = int(str(realign))
            align[realign_index, relation] = 1
    return align



class Vocab(object):
    def __init__(self, vocfile):
        self._word2id = {}
        self._id2word = []
        with open(vocfile, 'r') as f:
            index = 0
            for line in f:
                parts = line.split()
                id = 0
                if len(parts) == 2:
                    id = int(parts[1])
                elif len(parts) == 1:
                    id = index
                    index += 1
                else:
                    print('illegal voc line %s' % line)
                    continue
                self._word2id[parts[0]] = id
                self._id2word.append(parts[0])

    def get_voc_size(self):
        return len(self._id2word)

    def get_id(self, w):
        if not w in self._word2id:
            return self._word2id['<unk>']
        return self._word2id[w]

    def get_word(self, wid):
        if wid < 0 or wid >= len(self._id2word):
            return '<unk>'
        return self._id2word[wid]

    def get_eos(self):
        return self._word2id['</s>']

    def get_sos(self):
        return self._word2id['<s>']


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            if not isparent:
                child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3], True])
            else:
                child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3], False])
            isparent = True
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['Above', 'Below']:
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['Sup']:
                    return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
                elif child_list[i][2] not in ['Right', 'Above', 'Below', 'Sup']:
                    return_string += ['illegal']
        elif gtd_list[nodeid][0] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                     '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix']:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):

                if child_list[i][2] in ['Rstart']:
                    if child_list[i][3]:
                        return_string += convert(child_list[i][1], gtd_list)
                    else:
                        return_string += ['\\\\'] + convert(child_list[i][1], gtd_list)
                elif child_list[i][2] == 'Right':
                    if gtd_list[nodeid][0] in ['\\begincases']:
                        return_string += ['\\end' + gtd_list[nodeid][0][6:]] + convert(child_list[i][1], gtd_list)
                    else:
                        return_string += convert(child_list[i][1], gtd_list)
                        # elif child_list[i][2] not in ['Right', 'Above', 'Below']:
                        #     return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Leftsup':
                    return_string += ['\\['] + convert(child_list[i][1], gtd_list) + ['\\]']
                elif child_list[i][2] in ['Inside', 'boxed', 'textcircled']:
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['Sub', 'Below']:
                    if gtd_list[nodeid][0] in ['\\underline', '\\xrightarrow', '\\underrightarrow', '\\underbrace'] and \
                                    child_list[i][2] in ['Below']:
                        return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                    else:
                        return_string += ['_', '{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['Sup', 'Above']:
                    if gtd_list[nodeid][0] in ['\\overline', '\\widehat', '\\hat', '\\widetilde',
                                               '\\dot', '\\oversetfrown', '\\overrightarrow'] and child_list[i][2] in [
                        'Above']:
                        return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                    else:
                        return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['\\Nextline']:
                    return_string += ['\\\\'] + convert(child_list[i][1], gtd_list)
                elif child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)

        return return_string


def gtd2latex(cap):
    try:
        gtd_list = []
        gtd_list.append(['<s>', 0, -1, 'root'])
        for i in range(len(cap)):
            parts = cap[i]
            sym = parts[0]
            childid = int(parts[1])
            parentid = int(parts[3])
            relation = parts[4]
            gtd_list.append([sym, childid, parentid, relation])
        bool_endcases = False

        idx = -1
        for i in range(len(gtd_list)):
            if gtd_list[i][0] == '\\begincases':
                idx = i
        if idx != -1:
            bool_endcases = True
            for i in range(idx + 1, len(gtd_list)):
                if gtd_list[i][2] == idx and gtd_list[i][3] == 'Right':
                    bool_endcases = False

        latex_list = convert(1, gtd_list)
        if bool_endcases:
            latex_list += ['\\endcases']

        latex_list = np.array(latex_list)
        latex_list[latex_list == '\\space'] = '&'
        latex_list = list(latex_list)
        if 'illegal' in latex_list:
            latex_string = 'error3*3'
        else:
            latex_string = ' '.join(latex_list)
        return latex_string
    except:
        return ('error3*3')
        pass




def latex2gtd(cap):
    # cap = parts[1:]
    try:
        gtd_label = []
        # cap = '\\begincases a \\\\ b \\\\ c \\endcases = 1'
        # cap = '\\begincases  a + b  \\\\ c + d \\\\  e + f \\endcases = 1'
        cap = cap.split()
        gtd_stack = []
        idx = 0
        outidx = 1
        error_flag = False
        while idx < len(cap):
            # if idx == 16:
            #     print cap[idx]
            #     pdb.set_trace()
            if idx == 0:
                if cap[0] in ['{', '}']:
                    return ('error2*2: {} should NOT appears at START')

                if cap[0] not in ['\\beginaligned']:
                    string = cap[0] + '\t' + str(outidx) + '\t<s>\t0\tStart'
                    gtd_label.append(string.split('\t'))
                    outidx += 1
                else:
                    gtd_stack.append([cap[idx], str(outidx), 'Align', True])
                    idx += 1
                    string = cap[idx] + '\t' + str(outidx) + '\t<s>\t0\tStart'
                    gtd_label.append(string.split('\t'))
                    outidx += 1

                idx += 1

            else:
                # print(cap[idx])
                # pdb.set_trace()
                if cap[idx] == '{':
                    if cap[idx - 1] == '{':
                        return ('error2*2: double { appears')

                    elif cap[idx - 1] == '}' and gtd_stack:
                        if gtd_stack[-1][0] != '\\frac':
                            return ('error2*2: } { not follows frac ...')
                        else:
                            gtd_stack[-1][2] = 'Below'
                            idx += 1
                    else:
                        if cap[idx - 1] in ['\\frac', '\\overline', '\\widehat', '\\hat',
                                            '\\widetilde', '\\dot', '\\oversetfrown', '\\overrightarrow']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Above', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\sqrt']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Inside', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\underline', '\\xrightarrow', '\\underrightarrow', '\\underbrace']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Below', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\boxed', '\\textcircled']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Inside', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\bcancel']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Insert', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\begincases']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Rstart', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\\\']:
                            idx += 1
                        elif cap[idx - 1] == '_':
                            if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                return ('error2*2: ^ _ follows wrong math symbols')
                            # elif cap[idx - 2] in ['\\sum', '\\int', '\\lim', '\\bigcup', '\\bigcap']:
                            elif cap[idx - 2] in ['\\lim', '\\bigcup', '\\bigcap']:
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Below', True])
                                idx += 1
                            # elif gtd_stack and gtd_stack[-1][0] in ['\\sum', '\\int', '\\lim']:
                            elif gtd_stack and gtd_stack[-1][0] in ['\\lim']:
                                if gtd_stack[-1][2] != 'Below' and gtd_stack[-1][3]:
                                    gtd_stack[-1][2] = 'Below'
                                    gtd_stack[-1][3] = False
                                else:
                                    gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sub', True])
                                idx += 1
                            elif cap[idx - 2] == '}' and gtd_stack:
                                gtd_stack[-1][2] = 'Sub'
                                idx += 1
                            else:
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sub', True])
                                idx += 1
                        elif cap[idx - 1] == '^':

                            if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                return ('error2*2: ^ _ follows wrong math symbols')
                            # elif cap[idx - 2] in ['\\sum', '\\int', '\\lim']:  # 只能先尝试把int删掉了
                            elif cap[idx - 2] in ['\\lim']:  # 只能先尝试把int删掉了
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Above', True])
                                idx += 1
                            # elif gtd_stack and gtd_stack[-1][0] in ['\\sum', '\\int', '\\lim'] and cap[idx - 2] == '}':
                            elif gtd_stack and gtd_stack[-1][0] in ['\\lim'] and cap[idx - 2] == '}':
                                if gtd_stack[-1][2] != 'Above' and gtd_stack[-1][3]:
                                    gtd_stack[-1][2] = 'Above'
                                    gtd_stack[-1][3] = False
                                else:
                                    gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sup', True])
                                idx += 1
                            elif cap[idx - 2] == '}' and gtd_stack:
                                gtd_stack[-1][2] = 'Sup'
                                idx += 1
                            else:
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sup', True])
                                idx += 1
                        elif cap[idx - 1] == ']':
                            if gtd_stack and gtd_stack[-1][0] == '\\sqrt' and gtd_stack[-1][3]:
                                gtd_stack[-1][2] = 'Inside'
                                idx += 1
                                gtd_stack[-1][3] = False
                            else:
                                return ('error2*2: { follows unknown math symbols ...')
                        else:
                            return ('error2*2: { follows unknown math symbols ...')

                elif cap[idx] == '}':
                    if cap[idx - 1] in ['}', '\\endcases', '\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                        '\\endvmatrix', '\\endVmatrix'] and gtd_stack:
                        del (gtd_stack[-1])
                    idx += 1
                elif cap[idx] in ['\\endcases', '\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                  '\\endvmatrix', '\\endVmatrix']:
                    if cap[idx - 1] in ['\\endcases'] and gtd_stack:
                        del (gtd_stack[-1])
                    elif cap[idx - 1] in ['\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix', '\\endvmatrix',
                                          '\\endVmatrix'] and gtd_stack:
                        string = cap[idx - 1] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        del (gtd_stack[-1])
                        outidx += 1

                    if idx == len(cap) - 1 and cap[idx] not in ['\\endcases'] and gtd_stack:
                        string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        del (gtd_stack[-1])
                        outidx += 1
                    idx += 1
                elif cap[idx] in ['\\\\']:
                    if cap[idx-1] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                        '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix']:
                        return ('error2*2')
                    idx += 1
                elif cap[idx] in [']'] and gtd_stack and cap[idx - 1] in ['}'] and len(gtd_stack) > 1 and gtd_stack[-2][
                    0] == '\\sqrt' and gtd_stack[-2][2] == 'Leftsup':
                    del (gtd_stack[-1])
                    idx += 1
                elif cap[idx] in ['_', '^']:
                    if idx == len(cap) - 1:
                        return ('error2*2: ^ _ appers at end ...')
                    if cap[idx + 1] != '{':
                        return ('error2*2: ^ _ not follows { ...')
                    else:
                        idx += 1
                elif cap[idx] in ['\limits']:
                    return ('error2*2: \limits happens')

                elif cap[idx] == '[' and cap[idx - 1] != '\\\\' and cap[idx -1] not in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                        '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix'] and not (cap[idx - 1] in ['{'] and gtd_stack):
                    if cap[idx - 1] == '\\sqrt':
                        if cap[idx + 1] != ']':
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Leftsup', True])
                        else:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Leftsup', True])
                            idx += 1
                    elif cap[idx - 1] == '}' and gtd_stack:
                        string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        del (gtd_stack[-1])
                    else:
                        parts = string.split('\t')
                        string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                    idx += 1

                # elif idx == len(cap) - 1 and cap[idx] == ']' and not gtd_stack:
                #
                else:
                    if cap[idx - 1] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                        '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix']:

                        gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Rstart', True])

                    if cap[idx - 1] == '{' or (
                                            cap[idx - 1] == '[' and gtd_stack and gtd_stack[-1][0] == '\\sqrt' and
                                    gtd_stack[-1][
                                        2] == 'Leftsup') or cap[idx - 1] in ['\\begincases', '\\\\', '\\beginmatrix',
                                                                             '\\beginpmatrix',
                                                                             '\\beginbmatrix', '\\beginBmatrix',
                                                                             '\\beginvmatrix',
                                                                             '\\beginVmatrix']:
                        if cap[idx - 1] == '\\\\' and cap[idx - 2] == '}' and gtd_stack:
                            del (gtd_stack[-1])
                        if cap[idx - 1] == '\\\\' and gtd_stack == []:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + cap[idx - 2] + '\t' + \
                                     str(outidx - 1) + '\t\\Nextline'
                        elif cap[idx - 1] == '\\\\' and (
                                    gtd_stack and gtd_stack[-1][0] not in ['\\begincases', '\\beginmatrix',
                                                                           '\\beginpmatrix',
                                                                           '\\beginbmatrix', '\\beginBmatrix',
                                                                           '\\beginvmatrix',
                                                                           '\\beginVmatrix']):
                            string = cap[idx] + '\t' + str(outidx) + '\t' + cap[idx - 2] + '\t' + \
                                     str(outidx - 1) + '\t\\Nextline'
                        elif gtd_stack:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + \
                                     gtd_stack[-1][1] + '\t' + gtd_stack[-1][2]

                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
                    elif cap[idx - 1] == '}' and gtd_stack:
                        if cap[idx] not in ['\\endcases', '\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                            '\\endvmatrix', '\\endVmatrix']:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                                1] + '\tRight'
                            gtd_label.append(string.split('\t'))
                            outidx += 1
                            del (gtd_stack[-1])
                        idx += 1
                    elif cap[idx] == ']' and (gtd_stack and gtd_stack[-1][0] == '\\sqrt' and gtd_stack[-1][2] == 'Leftsup'):
                        idx += 1
                    elif cap[idx - 1] in ['\\endcases']:
                        while gtd_stack and gtd_stack[-1][0] != '\\begincases':
                            del (gtd_stack[-1])
                        string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
                        del (gtd_stack[-1])
                    elif gtd_stack and cap[idx - 1] in ['\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                                        '\\endvmatrix', '\\endVmatrix']:
                        string = cap[idx - 1] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1

                        parts = string.split('\t')
                        string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + \
                                 parts[1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
                        del (gtd_stack[-1])
                    else:
                        parts = string.split('\t')
                        if cap[idx] == '&' and (
                                    gtd_stack and gtd_stack[-1][0] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix',
                                                                       '\\beginbmatrix', '\\beginBmatrix', '\\beginvmatrix',
                                                                       '\\beginVmatrix']):
                            string = '\space' + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[1] + '\tRight'
                        else:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[
                                1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
        parts = string.split('\t')
        string = '</s>\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[1] + '\tEnd'
        gtd_label.append(string.split('\t'))
        return gtd_label
    except:
        return ('error2*2')
        pass

def save_gtd_label():

    bfs1_path = 'CROHME/'
    gtd_root_path = ''
    gtd_paths = ['train_chb','14_chb_test','16_chb_test','19_chb_test']

    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_label_gtd.pkl'
        out_label_fp = open(outpkl_label_file, 'wb')
        label_lines = {}
        process_num = 0
        origin_label = bfs1_path + gtd_path + '_labels.txt'
        
        with open(origin_label) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                key = line[0]
                latex = " ".join(i for i in line[1:])
                gtd = latex2gtd(latex)
                # print(key)
                # print(gtd)
                label_lines[key] = gtd

        

        print ('process files number ', process_num)

        pkl.dump(label_lines, out_label_fp)
        print ('save file done')
        out_label_fp.close()

def save_gtd_bidirecion_label():

    bfs1_path = 'CROHME/'
    gtd_root_path = ''
    gtd_paths = ['train_chb','14_chb_test','16_chb_test','19_chb_test']

    from utils import load_dict
    dict = load_dict(bfs1_path + 'dictionary_relation_9.txt')

    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_bidirection_label_gtd.pkl'
        out_label_fp = open(outpkl_label_file, 'wb')
        label_lines = {}
        process_num = 0
        origin_label = bfs1_path + gtd_path + '_labels.txt'
        
        with open(origin_label) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                key = line[0]
                latex = " ".join(i for i in line[1:])
                #latex = line[1]
                gtd = latex2gtd(latex)
                gtd_reverse = reverse_ver_3(gtd, dict)
                # print(key)
                # print(gtd)
                label_lines[key] = [gtd, gtd_reverse]

        

        print ('process files number ', process_num)

        pkl.dump(label_lines, out_label_fp)
        print ('save file done')
        out_label_fp.close()
def save_gtd_align():

    bfs1_path = 'CROHME/'
    gtd_root_path = ''
    gtd_paths = ['train_chb','14_chb_test','16_chb_test','19_chb_test']

    from utils import load_dict
    dict = load_dict(bfs1_path + 'dictionary_relation_9.txt')
    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_relations.pkl'
        
        out_label_fp = open(outpkl_label_file, 'wb')
        label_aligns = {}
        process_num = 0
        gtd_label = bfs1_path + gtd_path + '_label_gtd.pkl'
        fp_label=open(gtd_label, 'rb')
        gtds=pkl.load(fp_label)
        for uid, label_lines in gtds.items():
            # print(uid)
            # print(label_lines)
            align = gen_gtd_relation_align(label_lines,dict)
            label_aligns[uid] = align
            # print(align)
            # break
            

        print ('process files number ', process_num)

        pkl.dump(label_aligns, out_label_fp)
        print ('save file done')
        out_label_fp.close() 

def save_gtd_bidirection_align():

    bfs1_path = 'CROHME/'
    gtd_root_path = ''
    gtd_paths = ['train_chb','14_chb_test','16_chb_test','19_chb_test']

    from utils import load_dict
    dict = load_dict(bfs1_path + 'dictionary_relation_9.txt')
    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_bidirection_relations.pkl'
        
        out_label_fp = open(outpkl_label_file, 'wb')
        label_aligns = {}
        process_num = 0
        gtd_label = bfs1_path + gtd_path + '_bidirection_label_gtd.pkl'
        fp_label=open(gtd_label, 'rb')
        gtds=pkl.load(fp_label)
        for uid, label_lines in gtds.items():
            # print(uid)
            # print(label_lines)
            gtd, gtd_reverse = label_lines
            align = gen_gtd_relation_align(gtd,dict)
            align_reverse = gen_gtd_relation_align(gtd_reverse, dict)
            label_aligns[uid] = [align,align_reverse]
            # print(align)
            # break
            

        print ('process files number ', process_num)

        pkl.dump(label_aligns, out_label_fp)
        print ('save file done')
        out_label_fp.close() 

def reverse(gtd):
    import copy
    new_gtd = []
    max_id = 0
    for g in gtd:
        if(int(g[1]) > max_id):
            max_id = int(g[1])
    for i in range(len(gtd)):
        g = copy.deepcopy(gtd[i])
        # print(g)
        g[1] = str(int(g[1]))
        g[3] = str(int(g[3]))
        if(g[-1]=='Right'):
            tmp_id = g[1]
            g[1] = g[3]
            g[3] = tmp_id

            tmp_char = g[0]
            g[0] = g[2]
            g[2] = tmp_char
            new_gtd = [g] + new_gtd
        elif(g[-1] == 'End'):
            g[2] = gtd[i-1][2]
            g[3] = str(int(gtd[i-1][3]))

            tmp_id = g[1]
            g[1] = g[3]
            g[3] = tmp_id

            tmp_char = g[0]
            g[0] = g[2]
            g[2] = tmp_char

            g[-1] = 'Start'
            

            g[2] = '<s>'
            new_gtd = [g] + new_gtd
        elif(g[-1] == 'Start'):
            g[0] = gtd[i+1][2]
            g[1] = str(int(gtd[i+1][3]))

            tmp_id = g[1]
            g[1] = g[3]
            g[3] = tmp_id

            tmp_char = g[0]
            g[0] = g[2]
            g[2] = tmp_char

            g[-1] = 'End'
            g[0] = '</s>'
            new_gtd = [g] + new_gtd
        else:
            new_gtd = [g] + new_gtd
    # print(new_gtd)

    # reverse the id（test）
    id_trans_dict = {}
    id_trans_dict[0] = 0
    id_init = 1
    for g in new_gtd:
        id_trans_dict[int(g[1])] = id_init
        g[1] = str(id_init)
        id_init += 1
    for g in new_gtd:
        try:
            g[3] = str(id_trans_dict[int(g[3])])
        except:
            g[3] = str(0)


    return new_gtd

# def reverse(stack ,max_id):  # 写成递归的形式
#     if(stack == []):
#         return []
#     if(stack[-1][-1] == 'Right' or stack[-1][-1] == 'End' or stack[-1][-1] == 'Start'):
#         g = stack[-1]
#         stack.pop()
        
#         tmp_id = g[1]
#         g[1] = str(max_id-int(g[3]))
#         g[3] = str(max_id-int(tmp_id))

#         tmp_char = g[0]
#         g[0] = g[2]
#         g[2] = tmp_char

#         if(g[-1] == 'End'):
#             g[2] = '<s>'
#             g[-1] = 'Start'
#         elif(g[-1] == 'Start'):
#             g[-1] = 'End'
#             g[0] = '</s>'
#         return [g] + reverse(stack, max_id)

def reverse_rec(gtd, st, ed, parent):
    backbone = []
    # backbone_id = []
    new_gtd = []
    st_char = parent[0]
    st_id = int(parent[1])
    print(st_char, st_id)
    for g in gtd[st:ed+1]:
        if(int(g[-2]) == st_id and g[-1] == 'Right'):
            backbone = [g] + backbone
            st_id = int(g[1])
            st_char = g[0]

            
    if(len(backbone) <= 1):
        return reverse(gtd[st:ed+1])
    # print(backbone)
    backbone_idx = 0
    while(backbone_idx < len(backbone)):
        g = copy.deepcopy(backbone[backbone_idx])

        idx = g[1]

        tmp_id = g[1]
        g[1] = g[3]
        g[3] = tmp_id

        tmp_char = g[0]
        g[0] = g[2]
        g[2] = tmp_char

        new_gtd = [g] + new_gtd
        backbone_idx += 1

        for i in range(ed,st-1,-1):
            if(gtd[i][-2] == idx and (gtd[i] not in backbone)):
                new_ed = i
                while(i-1>=st and gtd[i-1][-2] != idx):
                    i -= 1

                new_st = i
                print("range")
                print(new_st, new_ed)
                new_gtd = reverse_rec(gtd, new_st, new_ed, backbone[backbone_idx]) + new_gtd 





    return new_gtd


def reverse_ver_2(gtd):
    new_gtd = reverse_rec(gtd, 1, len(gtd)-2, gtd[0])

    return new_gtd


# def test_reverse_2():

def deep_pri(graph, p):  # 深度遍历
    len = graph.shape[0]
    vis_list = []
    r_list = []
    for i in range(len):
        if(graph[p][i] == 7):  # 7是right
            r_list = [i] + r_list 
        elif(graph[p][i] != 0):
            vis_list =  vis_list + [i]
    vis_list = vis_list + r_list

    if(vis_list == []):
        return []

    res = []
    for item in vis_list:
        res += [[item, p, int(graph[p][item])]] + deep_pri(graph, item)
    return res 

def find_left_most(graph, p):  # 寻找最左端的字符
    len = graph.shape[0]
    for i in range(len):
        if(graph[p][i] == 7):  # 修改后right应该是7
            return find_left_most(graph, i)

    return p

# 这是一个目前有效的方案，前面的两个有错
def reverse_ver_3(gtd, relation_dict):  

    backbone = []
    new_gtd = []
    st_char = gtd[0][0]
    st_id = int(gtd[0][1])
    for g in gtd:
        if(int(g[-2]) == st_id and g[-1] == 'Right'):
            backbone = [g] + backbone
            st_id = int(g[1])
            st_char = g[0]


    char_num = len(gtd)
    relation_graph = np.zeros((char_num, char_num))
    tmp_relation_graph = np.zeros((char_num, char_num))
    char_dict = {}
    char_dict[int(gtd[0][1])] = gtd[0][0]
    rev_re_dict ={}
    for item in relation_dict.keys():
        rev_re_dict[relation_dict[item]] = item

    
    for g in gtd:
        if(g[-1] == 'Start' or g[-1] == 'End'):
            continue

        a_id = int(g[1])
        b_id = int(g[3])

        char_dict[a_id] = g[0]   #b -> a
        char_dict[b_id] = g[2]

        re_id = relation_dict[g[-1]]

        if(g[-1] == 'Right'):
            tmp_relation_graph[b_id][a_id] = re_id  # 

    
    for g in gtd:
        if(g[-1] == 'Start' or g[-1] == 'End'):
            continue

        a_id = int(g[1])
        b_id = int(g[3])

        char_dict[a_id] = g[0]   #b -> a
        char_dict[b_id] = g[2]

        re_id = relation_dict[g[-1]]

        if(g[-1] == 'Right'):
            relation_graph[a_id][b_id] = re_id  #  a->b
        else:
            a_id = find_left_most(tmp_relation_graph, a_id)
            relation_graph[b_id][a_id] = re_id # b->a

    if(backbone):
        st = int(backbone[0][1])
        ed = int(backbone[-1][3])
    else:
        st = st_id
        ed = st_id


    new_gtd += [[char_dict[st], str(st), '<s>' , '0', 'Start']]

    res = deep_pri(relation_graph, st)
    for item in res:
        p = int(item[1])
        c = item [0]
        re = item[2]

        new_gtd += [[char_dict[c], str(c), char_dict[p], str(p), rev_re_dict[re]]]



    new_gtd += [['</s>', str(len(gtd)), char_dict[ed], str(ed), 'End']]
        
    return new_gtd

    
def re_id(gtd):
    id_trans_dict = {}
    id_trans_dict[0] = 0
    id_init = 1
    for g in gtd:
        id_trans_dict[int(g[1])] = id_init
        g[1] = str(id_init)
        id_init += 1
    for g in gtd:
        try:
            g[3] = str(id_trans_dict[int(g[3])])
        except:
            g[3] = str(0)

    return gtd





            
if __name__ == '__main__':
    from utils import load_dict
    dict = load_dict('./dictionary_relation_9.txt')
    #latex = '\\sum _ { i = 1 } ^ { H }  \\sum _ { j = 1 } ^ { W }  \\sum _ { k = 1 } ^ { N ^ { c o l } }  \\frac { L (  F  ^ { c o l } _ {  i , j , k } ,   F  ^ { c o l } _ {  i , j , k } ) } { \\sum _ { i = 1 } ^ { H } \\sum _ { j = 1 } ^ { W }  \\sum _ { k = 1 } ^ { N ^ { c o l } }   F  ^ { c o l } _ {  i , j , k } }'
    latex = 'A _ { 2 k } = \\frac { 2 R A _ { k } } { 2 R + \sqrt { 4 R ^ { 2 } + A ^ { 2 } _ { k } } }'
    latex = '\sqrt [ 5 ] { 5 5 }'
    latex = '\\frac { 3 } { 4 }'
    # latex = '\\frac { \sqrt { A _ { k } ^ { 2 } + R ^ { 2 } 4 } + R 2 } { A _ { k } R 2 } = A _ { k 2 }'
    #latex = '- 7'
    # #latex = '\\frac { F _ { k , j , i } ^ { l o c } \\sum ^ { N ^ { l o c } } _ { 1 = k } \\sum ^ { W } _ { 1 = j } \\sum ^ { H } _ { 1 = i } } { ) F _ { k , j , i } ^ { l o c } , F _ { k , j , i } ^ { l o c } ( L } \\sum ^ { N ^ { l o c } } _ { 1 = k } \\sum ^ { W } _ { 1 = j } \\sum ^ { H } _ { 1 = i }'
    # # latex = 'a _ { i + j } ^ { k + p }'
    # latex = 'f ( x ) = \\sum _ { n = - \\infty } ^ { \\infty } a _ { n } = x ^ { \\alpha _ { n } }'
    # gtd = [['\\frac', '1', '<s>', '0', 'Start'], ['4', '2', '\\frac', '1', 'Above'], ['3', '3', '\\frac', '1', 'Below'], ['</s>', '4', '\\frac', '1', 'End']]
    # print(gtd)
    # print(gtd2latex(gtd[:-1]))

    gtd = latex2gtd(latex)
    print(gtd)
    print(gtd2latex(gtd))
    gtd = reverse_ver_3(gtd, dict)
    gtd = re_id(gtd)

    print(gtd)
    reverse = gtd2latex(gtd[:-1])
    print(reverse)
    # # gtd = latex2gtd(reverse)
    # # print(gtd2latex(gtd))
    # # gtd = reverse_ver_3(gtd, dict)
    # # gtd = re_id(gtd)
    # # # gtd = latex2gtd(latex)
    # print(gtd)
    # gtd = re_id(gtd)
    # rev_latex = gtd2latex(gtd)
    # print(rev_latex)
    
    # print(rev_latex)
    # rev_gtd = latex2gtd(rev_latex)
    # print(rev_gtd)
    # t = gen_gtd_relation_align(gtd, dict)
    # print(t)
    # gtd = reverse_ver_3(rev_gtd, dict)
    # print(gtd)
    # print(gtd2latex(gtd))
    # res = gtd2latex(gtd)
    # latex_reverse = gtd2latex(gtd_reverse)
    # print(t)

    # print(latex_reverse)
    # for item in gtd:
    #     print('\t\t'.join(item))
    # print('\n')

    # # print("reversed target: ")
    # # latext2 = '\\frac { y + x  } { w + z } + \\sqrt [ c ] { b } = a'
    # # gtd1 = latex2gtd(latext2)
    # # for item in gtd1:
    # #     print('\t\t'.join(item))


    # # latex ='\\sqrt [ c + d ] { b } + \\frac { y + x  } { w + z } = a'
    # # gtd = latex2gtd(latex)
    # # for item in gtd:
    # #     print('\t\t'.join(item))

    # print('\n')
    
    # # print(len(gtd)+1)
    # new_gtd = reverse_ver_3(copy.deepcopy(gtd),dict)
    # new_gtd = reverse_ver_3(copy.deepcopy(new_gtd))
    # new_gtd = re_id(new_gtd)
    # for item in new_gtd:
    #     print('\t\t'.join(item))

    # res = gtd2latex(new_gtd)

    # print('reverse')
    # print(res)

    



    
    
    # gtdr  = reverse(gtd)

    # res = gtd2latex(gtdr)
    # print(res)
    # for item in gtdr:
    #     print('\t\t'.join(item))
    
    # save_gtd_label()
    # save_gtd_align()
    # print(align)
    # print('\n')
    # print(gtd)
    # save_gtd_bidirecion_label()
    # save_gtd_bidirection_align()