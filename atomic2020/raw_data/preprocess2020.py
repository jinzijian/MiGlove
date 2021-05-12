# This Python file uses the following encoding: utf-8
import csv
data = []
with open('/p300/MiGlove/atomic2020/raw_data/train.tsv', encoding='utf-8') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    for line in tsvreader:
        data.append(line)
example = data[0]
sentence = ''
print(example[1])
for word in data[0]:
    sentence += word
    sentence += ' '
print(sentence)
train_sentences = []
event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
for i in range(len(data)):
    sentence = ''
    if(data[i][1] in event_center):
        for word in data[i]:
            sentence += word
            sentence += '\t'
        train_sentences.append(sentence)
    else:
        continue
print(train_sentences[:5])
fileObject = open('/p300/MiGlove/atomic2020/event_center/origin/2020train_split.txt','w', encoding='utf-8')
for word in train_sentences:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()



# This Python file uses the following encoding: utf-8
import csv
data = []
with open('/p300/MiGlove/atomic2020/raw_data/test.tsv', encoding='utf-8') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    for line in tsvreader:
        data.append(line)
example = data[0]
sentence = ''
print(example[1])
for word in data[0]:
    sentence += word
    sentence += ' '
print(sentence)
train_sentences = []
event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
for i in range(len(data)):
    sentence = ''
    if(data[i][1] in event_center):
        for word in data[i]:
            sentence += word
            sentence += '\t'
        train_sentences.append(sentence)
    else:
        continue
print(train_sentences[:5])
fileObject = open('/p300/MiGlove/atomic2020/event_center/origin/2020test_split.txt','w', encoding='utf-8')
for word in train_sentences:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()

# This Python file uses the following encoding: utf-8
import csv
data = []
with open('/p300/MiGlove/atomic2020/raw_data/dev.tsv', encoding='utf-8') as f:
    tsvreader = csv.reader(f, delimiter='\t')
    for line in tsvreader:
        data.append(line)
example = data[0]
sentence = ''
print(example[1])
for word in data[0]:
    sentence += word
    sentence += ' '
print(sentence)
train_sentences = []
event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
for i in range(len(data)):
    sentence = ''
    if(data[i][1] in event_center):
        for word in data[i]:
            sentence += word
            sentence += '\t'
        train_sentences.append(sentence)
    else:
        continue
print(train_sentences[:5])
fileObject = open('/p300/MiGlove/atomic2020/event_center/origin/2020dev_split.txt','w', encoding='utf-8')
for word in train_sentences:
    fileObject.write(word)
    fileObject.write('\n')
fileObject.close()