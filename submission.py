#*************************** IMPORTS ************************************
import pandas as pd
import build as b
import numpy as np
#import preprocessing as pp

from contest import contest as c

x = c("C:\Media\Coursera\data\sales_train_v2.csv", "C:/Media/Coursera/data/test.csv", 
      'C:/Media/Coursera/data/items.csv', 'C:/Media/Coursera/data/item_categories.csv')
x.GenerateSubmissionFiles(verbose=True)

#*************************** STATICS ************************************
outputFile = "C:/Media/Coursera/code/submission.csv"

#note: to create the pickle files, generate the build from the contest.py method GenerateSubmissionFiles() and then run the pickle.py

trainData = "C:/Media/Coursera/data/pp_data_sales_train_v2_2015.csv"
td_pickleFile = "C:/Media/Coursera/data/pp_data_sales_train_v2_pickle.pck"
trainLabels = "C:/Media/Coursera/data/pp_labels_sales_train_v2_2015.csv"
tl_pickleFile = "C:/Media/Coursera/data/pp_labels_sales_train_v2_pickle.pck"
testData = "C:/Media/Coursera/data/pp_test.csv"
tst_pickleFile = "C:/Media/Coursera/data/pp_test_pickle.pck"

#predictors = ['item_id','shop_id','date_block_num','item_price','item_category_id','new_item','new_shop_item']
label = 'item_cnt_month'

#*************************** MAIN ROUTINE ************************************
# Load the training data
#X = pd.read_csv(trainData,index_col=0)
#y = pd.read_csv(trainLabels,index_col=0)
#T = pd.read_csv(testData, index_col=0)

print("Opening training/test files...")
X = pd.read_pickle(td_pickleFile)
y = pd.read_pickle(tl_pickleFile)
T = pd.read_pickle(tst_pickleFile)

#X = b.downcast(X)
#y = b.downcast(y)
#T = b.downcast(T)
print("File opening complete!")

y.item_cnt_month = np.clip(y.item_cnt_month, 0, 20)
y = y[label]

restrict = ['item_cnt_month']
predictors = np.setdiff1d(np.sort(X.columns), np.sort(restrict))


#print("Building stacking CV regressor, 5 folds, full data set - XGB,LGBM,ET,MLP,AdaDTR with LR meta")
#model = b.build_StackingModelCV(posBias=False, cvRun=5)
#model.fit(X[predictors].as_matrix(), y.as_matrix())

#print("Building stacking regressor, LGBM + ETR + XGB + MLP with simple XGB meta")
#model = b.build_StackingModel()
#model.fit(X[predictors].as_matrix(), y.as_matrix())

print("Fitting a simple model (LGBM)")
model = b.build_SimpleModel()
model.fit(X[predictors], y)

#*************************** PRODUCE OUTPUT ************************************
print("Predicting new values...")
y = model.predict(T[predictors])
#y = model.predict(T[predictors].as_matrix())
T['item_cnt_month'] = y[:]

#all shipping sales go out of shop 12 (online store)
#T.loc[((T.item_id == 11365) & (T.shop_id != 12)), ['item_cnt_month']] = 0
Z = T[['item_cnt_month']].copy()
Z.item_cnt_month = np.clip(Z.item_cnt_month, 0, 20)
#Z.item_cnt_month = np.clip(Z.item_cnt_month, 0, 20)
Z.index.names = ['ID']

Z.to_csv(outputFile)
print ("File created!")
print ("\nTotal item_cnt_month estimated: {0}.".format(Z.item_cnt_month.sum()))

