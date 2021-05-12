import os
event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
with open('/p300/MiGlove/atomic2020/event_center/origin/2020train_split.txt',"r", encoding='utf-8') as f:
    test = f.readline()
    lines = f.readlines()

print(lines[0])
example = lines[0]
example = example.replace('PersonX', 'John')
print(example)
example = example.split('\t')
print(example[2])
new = []
for line in lines:
    line = line.lower()
    line = line.replace('person x', 'John')
    line = line.replace('personx', 'John')
    line = line.replace('person y', 'Tom')
    line = line.replace('persony', 'Tom')
    line = line.replace('person z', 'Jack')
    line = line.replace('personz', 'Jack')
    line = line.replace('isafter', 'after')
    line = line.replace('isbefore', 'before')
    line = line.replace('xreason', 'reason')
    line = line.replace('hassubevent', 'has sub event')
    line = line.replace('isfilledby', 'is filled by')
    line = line.replace('hinderedby', 'hindered by')
    
    new.append(line)

fileObject = open('/p300/MiGlove/atomic2020/event_center/forgraph/processed_train_split_graph1.txt','w', encoding='utf-8')
for line in new:
    fileObject.write(line)
fileObject.close()


event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
with open('/p300/MiGlove/atomic2020/event_center/origin/2020test_split.txt',"r", encoding='utf-8') as f:
    test = f.readline()
    lines = f.readlines()

print(lines[0])
example = lines[0]
example = example.replace('PersonX', 'John')
print(example)
example = example.split('\t')
print(example[2])
new = []
for line in lines:
    line = line.lower()
    line = line.replace('person x', 'John')
    line = line.replace('personx', 'John')
    line = line.replace('person y', 'Tom')
    line = line.replace('persony', 'Tom')
    line = line.replace('person z', 'Jack')
    line = line.replace('personz', 'Jack')
    line = line.replace('isafter', 'after')
    line = line.replace('isbefore', 'before')
    line = line.replace('xreason', 'reason')
    line = line.replace('hassubevent', 'has sub event')
    line = line.replace('isfilledby', 'is filled by')
    line = line.replace('hinderedby', 'hindered by')
    #line = line.replace('___', '')
    new.append(line)

fileObject = open('/p300/MiGlove/atomic2020/event_center/forgraph/processed_test_split_graph1.txt','w', encoding='utf-8')
for line in new:
    fileObject.write(line)
fileObject.close()

event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
with open('/p300/MiGlove/atomic2020/event_center/origin/2020dev_split.txt',"r", encoding='utf-8') as f:
    test = f.readline()
    lines = f.readlines()

print(lines[0])
example = lines[0]
example = example.replace('PersonX', 'John')
print(example)
example = example.split('\t')
print(example[2])
new = []
for line in lines:
    line = line.lower()
    line = line.replace('person x', 'John')
    line = line.replace('personx', 'John')
    line = line.replace('person y', 'Tom')
    line = line.replace('persony', 'Tom')
    line = line.replace('person z', 'Jack')
    line = line.replace('personz', 'Jack')
    line = line.replace('isafter', 'after')
    line = line.replace('isbefore', 'before')
    line = line.replace('xreason', 'reason')
    line = line.replace('hassubevent', 'has sub event')
    line = line.replace('isfilledby', 'is filled by')
    line = line.replace('hinderedby', 'hindered by')
    line = line.replace('___', '')
    new.append(line)

fileObject = open('/p300/MiGlove/atomic2020/event_center/forgraph/processed_dev_split_graph1.txt','w', encoding='utf-8')
for line in new:
    fileObject.write(line)
fileObject.close()