event_center = ['HinderedBy', 'isAfter', 'HasSubEvent', 'isBefore', 'xReason','Causes','isFilledBy']
with open('/p300/MiGlove/atomic2020/event_center/processed_train_split_graph.txt',"r", encoding='utf-8') as f:
    test = f.readline()
    lines = f.readlines()
