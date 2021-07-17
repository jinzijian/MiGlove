import os
train_path = '/p300/MiGlove/atomic2020/event_center/processed_train_split_graph.txt'
event_center = ['hindered by', 'after', 'has sub event', 'before', 'reason', 'causes', 'is filled by']
hinder = []
after = []
subevent = []
before = []
reason = []
causes = []
filled = []
with open(train_path,"r", encoding='utf-8') as f:
    test = f.readline()
    lines = f.readlines()
print(lines[:1])
for line in lines:
    line = line.split('\t')
    sentence = ''
    #print(line[1])
    if line[1] == 'hindered by':
        for word in line[:-1]:
            sentence += word
            sentence += '\t'
        hinder.append(sentence)
    if line[1] == 'after':
        for word in line[:-1]:
            sentence += word
            sentence += '\t'
        after.append(sentence)
    if line[1] == 'before':
        for word in line[:-1]:
            sentence += word
            sentence += '\t'
        before.append(sentence)
    if line[1] == 'causes':
        for word in line[:-1]:
            sentence += word
            sentence += '\t'
        causes.append(sentence)
    if line[1] == 'reason':
        for word in line[:-1]:
            sentence += word
            sentence += '\t'
        reason.append(sentence)
    if line[1] == 'has sub event':
        for word in line[:-1]:
            sentence += word
            sentence += '\t'
        subevent.append(sentence)
    if line[1] == 'is filled by':
        for word in line[:-1]:
            sentence += word
            sentence += '\t'
        filled.append(sentence)

fileObject = open('/p300/MiGlove/atomic2020/event_center/hinder.txt','w', encoding='utf-8')
for word in hinder:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()

fileObject = open('/p300/MiGlove/atomic2020/event_center/after.txt','w', encoding='utf-8')
for word in after:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()

fileObject = open('/p300/MiGlove/atomic2020/event_center/before.txt','w', encoding='utf-8')
for word in before:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()

fileObject = open('/p300/MiGlove/atomic2020/event_center/causes.txt','w', encoding='utf-8')
for word in causes:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()

fileObject = open('/p300/MiGlove/atomic2020/event_center/reason.txt','w', encoding='utf-8')
for word in reason:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()

fileObject = open('/p300/MiGlove/atomic2020/event_center/filled.txt','w', encoding='utf-8')
for word in filled:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()

fileObject = open('/p300/MiGlove/atomic2020/event_center/subevent.txt','w', encoding='utf-8')
for word in subevent:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()