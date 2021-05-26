#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 0:13 上午
# @Author  : Gear
# event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
def deal_sentence(line):
    if 'hindered by' in line[1]:
        p = line.index('hindered by')
        line[p] = 'is hindered by'
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'causes' in line[1]:
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'has sub event' in line[1]:
        p = line.index('has sub event')
        line[p] = 'has a sub event that'
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'after' in line[1]:
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'before' in line[1]:
        sentence = ''
        for word in line[:-1]:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'reason' in line[1]:
        p = line.index('reason')
        line[p] = 'because'
        sentence = ''
        for word in line:
            if(word =='\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

    if 'is filled by' in line[1]:
        # p = line.index('reason')
        line_tmp = line[0].split(' ')
        p = line_tmp.index('kkk')
        line_tmp[p] = line[-2]
        #print(line_tmp)
        sentence = ''
        for word in line_tmp:
            if (word == '\n'):
                continue
            sentence += word
            sentence += ' '
        sentence += '\n'
        return sentence

def equal_list(list1, list2):
    if(len(list1) != len(list2)):
        return False
    for i in range(len(list1)):
        if(list1[i] != list2[i]):
            return False
    return True

def find_idx(string_list, node_list):
    length = len(node_list)
    for i in range(len(string_list) - length + 1):
        sub = string_list[i:i+length]
        if(equal_list(sub, node_list)):
            return i
    return -1

if __name__ == '__main__':

    with open('/p300/MiGlove/atomic2020/event_center/processed_dev_split_graph.txt', "r", encoding='utf-8') as f:
        lines = f.readlines()
    sentences = []
    print(len(lines))
    for line in lines:
        line = line.split('\t')
        sentence = deal_sentence(line)
        sentences.append(sentence)
    print(len(sentences))
    file = open('/p300/MiGlove/atomic2020/event_center/bert_pretest.txt', 'w', encoding='utf-8')
    for sentence in sentences:
        if(sentence.count('\n') == 2):
            print(sentence)
        file.write(sentence)
    file.close()


    with open('/p300/MiGlove/atomic2020/event_center/processed_dev_split_graph.txt', "r", encoding='utf-8') as f:
        old_lines = f.readlines()
    with open('/p300/MiGlove/atomic2020/event_center/bert_pretest.txt', 'r', encoding='utf-8') as f:
        new_lines = f.readlines()
    print(len(old_lines))
    print(len(new_lines))
    print(old_lines[17313])
    print(new_lines[17313])
    old_line = old_lines[17313].split('\t')
    new_line = new_lines[17313].split(' ')
    #存储位置
    src = old_line[0].split(' ')
    tgt = old_line[1].split(' ')
    src_b = []
    src_e = []
    tgt_b = []
    tgt_e = []
    for i in range(len(old_lines)):
        old_line = old_lines[i].split('\t')
        new_line = new_lines[i].split(' ')
        sb = find_idx(new_line, old_line[0].split(' '))
        se = sb + len(old_line[0].split(' '))
        tb = find_idx(new_line, old_line[2].split(' '))
        te = tb + len(old_line[2].split(' '))
        src_b.append(sb)
        src_e.append(se)
        tgt_b.append(tb)
        tgt_e.append(te)
    #得到bert embbeddings
    