import pandas as pd

trainData = "C:/Media/Coursera/data/pp_data_sales_train_v2.csv"
td_pickleFile = "C:/Media/Coursera/data/pp_data_sales_train_v2_pickle.pck"
trainLabels = "C:/Media/Coursera/data/pp_labels_sales_train_v2.csv"
tl_pickleFile = "C:/Media/Coursera/data/pp_labels_sales_train_v2_pickle.pck"
testData = "C:/Media/Coursera/data/pp_test.csv"
tst_pickleFile = "C:/Media/Coursera/data/pp_test_pickle.pck"

print("opening files...")
td = pd.read_csv(trainData, index_col=0)
tl = pd.read_csv(trainLabels, index_col=0)
tst = pd.read_csv(testData, index_col=0)

print("files opened.")
print("pickling train data to '%s'..." % td_pickleFile )
td.to_pickle(td_pickleFile)

print("pickling train labels to '%s'..." % tl_pickleFile )
tl.to_pickle(tl_pickleFile)

print("pickling test data to '%s'..." % tst_pickleFile )
tst.to_pickle(tst_pickleFile)

print("Complete!")