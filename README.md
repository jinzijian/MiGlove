# MiGlove
 ### 流程

1. 创建图

2. 利用BERT求Node embedding

3. 求Graph embedding

4. 求互信息

### 待办事宜

- [x] 求graph emb使用GPU

- [ ] 提高Link predict的AUC

- [ ] 调求MI的参数

- [ ] 是不是要把三种求Graph embedding的方法的auc控制在一个值？

- [x] NWJ/NCE bug修复

- [x] bert emb转化位置

- [x] 求bert emb

- [x] bert emb和node id对应上



### 实验参数

|||||
|-|-|-|-|
|参数信息|graphsage|NMP|deepwalk|
|lr||||
|auc||||


|||||
|-|-|-|-|
|参数信息|mine|nwj|nce|
|lr||||
|auc||||

