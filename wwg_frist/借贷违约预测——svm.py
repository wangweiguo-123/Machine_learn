import csv
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb

feature=[]
#标签
lable=[]
#打开csv训练集文件
#获取csv文件句柄，汉字用utf8解码
f=open(r'C:\Users\WWG\Downloads\\train.csv',encoding='utf8')
#读取文件内容
f_csv=csv.reader(f)
head=next(f_csv)
for row in f_csv:
    temp=[]
    #1,2,3,4,5,7,9,10,17,18,21,22,24,25,28,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46
    column=[1,2,3,4,7,8,9,10,14,16,17,18,21,22,24,25,26,28,37,40,46]
    for col in column:
        if col==8:
            if row[8]=='':
                break
            digit=[]
            for word in row[8]:
                if word.isdigit():
                   digit.append(word)
            temp.append(eval(''.join(digit)))
            continue
        if row[col]=='':
            break
        temp.append(eval(row[col]))
    if len(temp)==21:
        feature.append(temp)
        lable.append(eval(row[13]))
f.close()

#引入测试集
feature_test=[]
f=open(r'C:\Users\WWG\Downloads\testA.csv',encoding='utf8')
#读取文件内容
f_csv=csv.reader(f)
head_test=next(f_csv)
for row_test in f_csv:
    temp=[]
    #,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46
    column=[1,2,3,4,7,8,9,10,13,15,16,17,20,21,23,24,25,27,36,39,45]
    #[14436, 3, 13, 438, 72435, 0, 76451, 1, 16, 18, 0, 11, 0, 16255, 51, 24, 0, 8, 14, 2]
    map_col={'1':14436,'2':3,'3':13,'4':438,'7':72435,'8':5,'9':0,'10':76451,'13':1,'15':16,'16':18,'17':0,'20':11,'21':0,'23':16255,'24':51,'25':24,'27':0,'36':8,'39':14,'45':2}
    for col in column:
        if col==8:
            digit=[]
            for word in row[8]:
                if word.isdigit():
                   digit.append(word)
                if len(digit)==0:
                    digit.append('2')
            temp.append(eval(''.join(digit)))
            continue
        if row_test[col]=='':
            temp.append(map_col[str(col)])
            continue
        temp.append(eval(row_test[col]))
    feature_test.append(temp)
f.close()
print('测试集数量：',len(feature_test))



feature_test=np.array(feature_test).reshape((len(feature_test),len(feature_test[0])))
feature=np.array(feature).reshape((len(feature),len(feature[0])))
lable=np.array(lable).reshape((len(lable),1))
train_x,test_x,train_y,test_y=train_test_split(feature,lable,test_size=0.3)
# model =xgboost.XGBClassifier(n_estimators=100,scale_pos_weight=5,max_depth=4,objective='binary:logitraw',eval_metric='auc')  0.6435      # 载入模型（模型命名为model)
# model =xgboost.XGBClassifier(n_estimators=100,scale_pos_weight=6,max_depth=6,objective='binary:logitraw',learning_rate=0.1) 0.6512s

eval_set=[(test_x,test_y)]
model =xgboost.XGBClassifier(learning_rate =0.009,
                n_estimators=100,
                gamma=1,
                max_depth=9,
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary:logistic',
                scale_pos_weight=4 )
model.fit(train_x,train_y,early_stopping_rounds=10, eval_set=eval_set,verbose=True,eval_metric = "error")            # 训练模型（训练集）
predict_test = list(model.predict(feature_test))
predict_=list(model.predict(test_x))
test_accuracy=accuracy_score(test_y,predict_)
print('准确率为：',test_accuracy)

#检查占比率
m=0
for t in predict_test:
    if t==1:
        m+=1
print('违约占比：',m/len(predict_test))


ID=[]
for i in range(200000):
    ID.append(800000+i)
datafarm=pd.DataFrame({'id':ID,'isDefault':predict_test})
datafarm.to_csv(r'predict.csv',sep=',',index=False)
print('finish')


