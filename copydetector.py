
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd   
import jieba 
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split



# 加载停用词
with open('chinese_stopwords.txt','r',encoding='utf-8') as file:
	stopwords = [i[:-1] for i in file.readlines()]

#加载数据
news = pd.read_csv('sqlResult_1558435.csv',encoding='gb18030')

#处理缺失值
news = news.dropna(subset=['content'])
print(news.shape)
print(news.head())

# 分词
def split_text(text):
	text = text.replace(' ','')
	text = text.replace('\n', '')
	text2 = jieba.cut(text.strip())
	result = ''.join([w for w in text2 if w not in stopwords])
	return result

if not os.path.exists('corpus.pkl'):
	corpus = list(map(split_text,[str(i) for i in news.content]))
	print(corpus[0])
	with open('corpus.pkl','wb') as file:
		pickle.dump(corpus,file)
else:
	with open('corpus.pkl','rb') as file:
		corpus = pickle.load(file)

#得到corpus 的TF-IDF 矩阵

countVectorizer =  CountVectorizer(encoding='gb18030',min_df=0.015)
tfidfTransformer = TfidfTransformer()
countvector = countVectorizer.fit_transform(corpus)
tfidf = tfidfTransformer.fit_transform(countvector)


# 标记是否为自己的新闻
label = list(map(lambda source:1 if '新华' in str(source) else 0, news.source))
#数据集切分
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size = 0.3) 


clf =MultinomialNB()
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

#使用模型检测抄袭新闻，预测风格
prediction = clf.predict(tfidf.toarray())
labels = np.array(label)
compare_news_index = pd.DataFrame({'prediction': prediction, 'labels' :labels})
copy_news_index =compare_news_index[(compare_news_index['prediction']==1) & (compare_news_index['labels']==0)]
#实际为新华社的新闻
xinhuashe_news_index = compare_news_index[(compare_news_index['labels']==1)].index
 
print('可能为copy的新闻条数', len(copy_news_index))

print(copy_news_index)

from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
scaled_array = normalizer.fit_transform(tfidf.toarray())

from sklearn.cluster import KMeans

#使用全量文档进行聚类

kmeans = KMeans(n_clusters =25)
k_labels = kmeans.fit_predict(scaled_array)

with open('label.pkl','wb') as file:
	pickle.dump(k_labels,file)

id_class = {index:class_ for index,class_ in enumerate(k_labels)}
with open('id_class.pkl','wb') as file:
	pickle.dump(id_class,file)

from collections import defaultdict
class_id = defaultdict(set)
for index, class_ in id_class.items():
	# 只统计新华社发布class_id
	if index in xinhuashe_news_index.tolist():
		class_id[class_].add(index)
	with open('class_id.pkl','wb') as file:
		pickle.dump(class_id,file)

#查找相似文本

def find_similar_text(cpindex, top = 10):
	dist_dict = {i:cosine_similarity(tfidf[cpindex],tfidf[i]) for i in class_id[id_class[cpindex]]}

	return sorted(dist_dict.items(),key = lambda x:x[1][0],reverse = True)[:top]

#在 copy_News_Inded里面查找一个

cpindex = 2253
similar_list = find_similar_text(cpindex)
print(similar_list)
print('怀疑抄袭\n',news.iloc[cpindex].content)

# 找相似原文
similar2 = similar_list[0][0]
print('相似原文\n', news.iloc[similar2].content)






