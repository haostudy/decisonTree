# -*- coding: UTF-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus

import os
os.environ["PATH"] += os.pathsep + 'C:/software/graphviz2.38/bin/'  #注意修改你的路径

if __name__ == '__main__':

	disk = []
	disk_consequence = []
	with open('Disk_SMART_dataset.txt', 'r') as fr:  # 加载文件
		for inst in fr.readlines():
			temp = inst.strip().split(', ')
			listTemp = []
			a = len(temp)
			for i in range(a):
				if(i == 6 or i == 7 or i == 9 or i == 3 or i == 5):#序号需要-1，数组从0开始
					listTemp.append(temp[i])
			listTemp.append(temp[1])
			disk.append(listTemp)
			# disk.append(inst.strip().split(', '))  # 处理文件

	disk_target = []														#提取每组数据的类别，保存在列表里
	for each in disk:
		disk_target.append(each[-1])
	print(disk_target)

	diskLabels = ['SUT', 'SER','POH', 'RUE', 'TC']			            #特征标签
	disk_list = []														#保存disk数据的临时列表
	disk_dict = {}														#保存disk数据的字典，用于生成pandas
	for each_label in diskLabels:											#提取信息，生成字典
		for each in disk:
			disk_list.append(each[diskLabels.index(each_label)])
		disk_dict[each_label] = disk_list
		disk_list = []
	print(disk_dict)														#打印字典信息
	disk_pd = pd.DataFrame(disk_dict)									#生成pandas.DataFrame
	print(disk_pd)														#打印pandas.DataFrame
	le = LabelEncoder()														#创建LabelEncoder()对象，用于序列化			
	for col in disk_pd.columns:											#序列化
		disk_pd[col] = le.fit_transform(disk_pd[col])
	print(disk_pd)														#打印编码信息

	clf = tree.DecisionTreeClassifier(max_depth = 4)						#创建DecisionTreeClassifier()类
	clf = clf.fit(disk_pd.values.tolist(), disk_target)					#使用数据，构建决策树

	dot_data = StringIO()
	tree.export_graphviz(clf, out_file = dot_data,							#绘制决策树
						feature_names = disk_pd.keys(),
						class_names = clf.classes_,
						filled=True, rounded=True,
						special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf("diskTree.pdf")												#保存绘制好的决策树，以PDF的形式存储。

	print(clf.predict([[1,1,1,0]]))											#预测
