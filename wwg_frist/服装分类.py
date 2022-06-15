from sklearn.metrics import accuracy_score
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
import pandas as pd

#读取训练数据集
data=pd.read_csv(r'C:\Users\WWG\Desktop\fashion-mnist_train.csv')
#放入panda的框架
data=pd.DataFrame(data=data)
#取出标签列
label=data['label'].values
#训练数据集删除label标签列
data.pop('label')
#训练数据集的feature特征属性值导出
feature=data.values
print('服装标签值为： ',label)

#筛选多余的特征属性列
sum=[]
for i in range(len(feature)):
    if i==0:
        for j in range(len(feature[0])):
            sum.append(feature[0][j])
    else:
        for k in range(len(feature[0])):
            sum[k]=sum[k]+feature[k][j]
print('每一属性列计数：',sum)
index=[]
for num in range(len(sum)):
    if sum[num]==0:
        index.append(num)
print('应该从测试集与训练集同时过滤掉的属性列号：',index)

#读取测试数据集
data_test=pd.read_csv(r'C:\Users\WWG\Desktop\fashion-mnist_test_data.csv')
data_test=pd.DataFrame(data=data_test)
#删除测试集中的图片编号ID列
data_test.pop('ID')

#特征删减
data.drop(data.columns[index],axis=1,inplace=True)
data_test.drop(data_test.columns[index],axis=1,inplace=True)
#过滤之后的特征重新导出
Test_Feature=data_test.values
feature=data.values
#numpy转化成矩阵形式
feature=np.array(feature).reshape((len(feature),len(feature[0])))
label=np.array(label).reshape((len(label),1))
train_x,test_x,train_y,test_y=train_test_split(feature,label,test_size=0.3)
print('开始训练....')




#XGBoost模型建立，设置学习率为0.1，训练500棵树，每棵树最深为10层，随机抽样百分之八十的样本进行训练，模式为多分类
model =xgboost.XGBClassifier(learning_rate =0.1,
                n_estimators=500,
                gamma=0.2,
                max_depth=10,
                subsample=0.8,
                colsample_bytree=0.5,
                objective= 'multi：softprob')
model.fit(train_x,train_y)            # 训练模型（训练集）
predict_test = list(model.predict(Test_Feature))
predict=list(model.predict(test_x))
test_accuracy=accuracy_score(test_y,predict)
print('准确率为：',test_accuracy)

# ID=[]
# for i in range(10000):
#     ID.append(str(i)+'.jpg')
# datafarm=pd.DataFrame({'id':ID,'label':predict_test})
# datafarm.to_csv(r'result.csv',sep=',',index=False)
# print('finish')
