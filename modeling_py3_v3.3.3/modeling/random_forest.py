import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def MakeupMissingCategorical(x):
    if str(x) == 'nan':
        return 'Unknown'
    else:
        return x

def MakeupMissingNumerical(x,replacement):
    if np.isnan(x):
        return replacement
    else:
        return x


'''
第一步：文件准备
'''
foldOfData = 'H:/'
mydata = pd.read_csv(foldOfData + "还款率模型.csv",header = 0,engine ='python')
#催收还款率等于催收金额/（所欠本息+催收费用）。其中催收费用以支出形式表示
mydata['rec_rate'] = mydata.apply(lambda x: x.LP_NonPrincipalRecoverypayments /(x.AmountDelinquent-x.LP_CollectionFees), axis=1)
#还款率假如大于1，按作1处理
mydata['rec_rate'] = mydata['rec_rate'].map(lambda x: min(x,1))
#整个开发数据分为训练集、测试集2个部分
trainData, testData = train_test_split(mydata,test_size=0.4)

'''
第二步：数据预处理
'''
#由于不存在数据字典，所以只分类了一些数据
categoricalFeatures = ['CreditGrade','Term','BorrowerState','Occupation','EmploymentStatus','IsBorrowerHomeowner','CurrentlyInGroup','IncomeVerifiable']

numFeatures = ['BorrowerAPR','BorrowerRate','LenderYield','ProsperRating (numeric)','ProsperScore','ListingCategory (numeric)','EmploymentStatusDuration','CurrentCreditLines',
                'OpenCreditLines','TotalCreditLinespast7years','CreditScoreRangeLower','OpenRevolvingAccounts','OpenRevolvingMonthlyPayment','InquiriesLast6Months','TotalInquiries',
               'CurrentDelinquencies','DelinquenciesLast7Years','PublicRecordsLast10Years','PublicRecordsLast12Months','BankcardUtilization','TradesNeverDelinquent (percentage)',
               'TradesOpenedLast6Months','DebtToIncomeRatio','LoanFirstDefaultedCycleNumber','LoanMonthsSinceOrigination','PercentFunded','Recommendations','InvestmentFromFriendsCount',
               'Investors']

'''
类别型变量需要用目标变量的均值进行编码
'''
encodedFeatures = []
encodedDict = {}
for var in categoricalFeatures:
    trainData[var] = trainData[var].map(MakeupMissingCategorical)
    avgTarget = trainData.groupby([var])['rec_rate'].mean()
    avgTarget = avgTarget.to_dict()
    newVar = var + '_encoded'
    trainData[newVar] = trainData[var].map(avgTarget)
    encodedFeatures.append(newVar)
    encodedDict[var] = avgTarget

#对数值型数据的缺失进行补缺
trainData['ProsperRating (numeric)'] = trainData['ProsperRating (numeric)'].map(lambda x: MakeupMissingNumerical(x,0))
trainData['ProsperScore'] = trainData['ProsperScore'].map(lambda x: MakeupMissingNumerical(x,0))

avgDebtToIncomeRatio = np.mean(trainData['DebtToIncomeRatio'])
trainData['DebtToIncomeRatio'] = trainData['DebtToIncomeRatio'].map(lambda x: MakeupMissingNumerical(x,avgDebtToIncomeRatio))
numFeatures2 = numFeatures + encodedFeatures

'''
第三步：调参
对基于CART的随机森林的调参，主要有：
1，树的个数
2，树的最大深度
3，内部节点最少样本数与叶节点最少样本数
4，特征个数
此外，调参过程中选择的误差函数是均值误差，5倍折叠
'''
X, y= trainData[numFeatures2],trainData['rec_rate']

param_test1 = {'n_estimators':range(60,91,5)}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(min_samples_split=50,min_samples_leaf=10,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='neg_mean_squared_error',cv=5)
gsearch1.fit(X,y)
gsearch1.best_params_, gsearch1.best_score_
best_n_estimators = gsearch1.best_params_['n_estimators']

param_test2 = {'max_depth':range(3,15), 'min_samples_split':range(10,101,10)}
gsearch2 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, min_samples_leaf=10,max_features='sqrt' ,random_state=10,oob_score=True),param_grid = param_test2, scoring='neg_mean_squared_error',cv=5)
gsearch2.fit(X,y)
gsearch2.best_params_, gsearch2.best_score_
best_max_depth = gsearch2.best_params_['max_depth']
best_min_samples_split = gsearch2.best_params_['min_samples_split']

param_test3 = {'min_samples_leaf':range(1,20,2)}
gsearch3 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, max_depth = best_max_depth,max_features='sqrt',min_samples_split=best_min_samples_split,random_state=10,oob_score=True),param_grid = param_test3, scoring='neg_mean_squared_error',cv=5)
gsearch3.fit(X,y)
gsearch3.best_params_, gsearch3.best_score_
best_min_samples_leaf = gsearch3.best_params_['min_samples_leaf']

numOfFeatures = len(numFeatures2)
mostSelectedFeatures = numOfFeatures/2
param_test4 = {'max_features':range(3,numOfFeatures+1)}
gsearch4 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf,min_samples_split=best_min_samples_split,random_state=10,oob_score=True),param_grid = param_test4, scoring='neg_mean_squared_error',cv=5)
gsearch4.fit(X,y)
gsearch4.best_params_, gsearch4.best_score_
best_max_features = gsearch4.best_params_['max_features']

#把最优参数全部获取去做随机森林拟合
cls = RandomForestRegressor(n_estimators=best_n_estimators,max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf,min_samples_split=best_min_samples_split,max_features=best_max_features,random_state=10,oob_score=True)
cls.fit(X,y)
trainData['pred'] = cls.predict(trainData[numFeatures2])
trainData['less_rr'] = trainData.apply(lambda x: int(x.pred > x.rec_rate), axis=1)
np.mean(trainData['less_rr'])
err = trainData.apply(lambda x: np.abs(x.pred - x.rec_rate), axis=1)
np.mean(err)

#随机森林评估变量重要性
importance=cls.feature_importances_
featureImportance=dict(zip(numFeatures2,importance))
featureImportance=sorted(featureImportance.items(),key=lambda x:x[1],reverse=True)

'''
第四步：在测试集上测试效果
'''
#类别型数据处理
for var in categoricalFeatures:
    testData[var] = testData[var].map(MakeupMissingCategorical)
    newVar = var + '_encoded'
    testData[newVar] = testData[var].map(encodedDict[var])
    avgnewVar = np.mean(trainData[newVar])
    testData[newVar] = testData[newVar].map(lambda x: MakeupMissingNumerical(x, avgnewVar))

#连续性数据处理
testData['ProsperRating (numeric)'] = testData['ProsperRating (numeric)'].map(lambda x: MakeupMissingNumerical(x,0))
testData['ProsperScore'] = testData['ProsperScore'].map(lambda x: MakeupMissingNumerical(x,0))
testData['DebtToIncomeRatio'] = testData['DebtToIncomeRatio'].map(lambda x: MakeupMissingNumerical(x,avgDebtToIncomeRatio))

testData['pred'] = cls.predict(testData[numFeatures2])
testData['less_rr'] = testData.apply(lambda x: int(x.pred > x.rec_rate), axis=1)
np.mean(testData['less_rr'])
err = testData.apply(lambda x: np.abs(x.pred - x.rec_rate), axis=1)
np.mean(err)