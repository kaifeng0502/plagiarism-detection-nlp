### To DO:
此项目一个抄袭自动检测分析的工具:

* 定义可能抄袭的文章来源
* 与原文对比定位抄袭的地方

原始数据：sqlResult.csv，共计89611篇
从数据库导出的文章，字段包括：id, author, source, content, feature, title, url
常用中文停用词：chinese_stopwords.txt

### 总体思路：

* 预测文章风格是否和自己一致 => 分类算法
* 根据模型预测的结果来对全量文本进行比对，如果数量很大，=> 可以先聚类降维，比如将全部文档自动聚成k=25类
* 文本特征提取 => 计算TF-IDF
* TopN相似 => TF-IDF相似度矩阵中TopN文档
* 编辑距离editdistance => 计算句子或文章之间的编辑距离


### 步骤：
文本抄袭自动检测：

* tep1，数据加载
加载sqlResult.csv及停用词chinese_stopwords.txt
* Step2，数据预处理
1）数据清洗，针对content字段为空的情况，进行dropna
2）分词，使用jieba进行分词
3）将处理好的分词保存到 corpus.pkl，方便下次调用
4）数据集切分，70%训练集，30%测试集
* Step3，提取文本特征TF-IDF
* Step4，预测文章风格是否和自己一致
使用分类模型（比如MultinomialNB），对于文本的特征（比如TF-IDF）和label（是否为新华社）进行训练
* Step5，找到可能Copy的文章，即预测label=1，但实际label=0
* Step6，根据模型预测的结果来对全量文本进行比对，如果数量很大，我们可以先用k-means进行聚类降维，比如k=25种聚类
* Step7，找到一篇可能的Copy文章，从相同label中，找到对应新华社的文章，并按照TF-IDF相似度矩阵，从大到小排序，取Top10
* Step8，使用编辑距离editdistance，计算两篇文章的距离
* Step9，精细比对，对于疑似文章与原文进行逐句比对，即计算每个句子的编辑距离editdistance


