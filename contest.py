import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils.extmath import cartesian
from itertools import product
from sklearn import preprocessing
import gc

class contest(object):
    
    __preferredColumnOrder = ['item_id','shop_id','date_block_num','quarter','half','year','item_category_id','new_item','new_shop_item',
                              'mode_item_price_month','min_item_price_month','max_item_price_month','mean_item_price_month',
                              'mean_item_category_price_month','min_item_category_price_month','max_item_category_price_month', 'mode_item_category_price_month']
    
    def __init__(self, trainDataFile, testDataFile, itemDataFile, categoryDataFile):
        
        #validate that files were passed in and exist at location provided by caller
        if (not trainDataFile) | (not testDataFile) | (not itemDataFile) | (not categoryDataFile):
            raise RuntimeError('file locations must be provided for train, test, items, and category data.')
        
        for i,x in [[trainDataFile,'Train'], [testDataFile,'Test'], [itemDataFile, 'Item'], [categoryDataFile, 'Category']]:
            i = str(i).replace('\\','/').strip()
            if not Path(i).is_file():
                raise RuntimeError('%s data file speicified [{%s}] does not exist.' % (x, i))
            if x == 'Train':
                self.__orig_trainDataFile = i
            elif x == 'Test':
                self.__orig_testDataFile = i
            elif x == 'Item':
                self.__orig_itemDataFile = i
            else:
                self.__orig_categoryDataFile = i
        
        self.__out_trainDataFile = self.__outputFile(self.__orig_trainDataFile, 'pp_data_')
        self.__out_trainLabelsFile = self.__outputFile(self.__orig_trainDataFile, 'pp_labels_')
        self.__out_testDataFile = self.__outputFile(self.__orig_testDataFile, 'pp_')
        
        self.__out_validateTrainDataFile = self.__outputFile(self.__orig_trainDataFile, 'val_train_data_')
        self.__out_validateTrainLabelsFile = self.__outputFile(self.__orig_trainDataFile, 'val_train_labels_')
        self.__out_validateTestDataFile = self.__outputFile(self.__orig_trainDataFile, 'val_test_data_')
        self.__out_validateTestLabelsFile = self.__outputFile(self.__orig_trainDataFile, 'val_test_labels_')
    
    def __outputFile(self, inFile, prefix):

        x = inFile.split('/')
        x[len(x) - 1] = prefix + x[len(x) - 1]
        x = "/".join(x)
        return x

    def __downcast(self, df):
        #reduce all float and int 64 values down to 32-bit to save memory
        floats = [c for c in df if df[c].dtype == 'float64']
        ints = [c for c in df if df[c].dtype == 'int64']
        df[floats] = df[floats].astype(np.float32)
        df[ints] = df[ints].astype(np.int32)
    
        return df

    def __openFilePrepared(self, fileName):
        #open all files with no pre-specified index; downcast numeric data from 64 to 32-bit
        df = pd.read_csv(fileName, index_col=False)
        df = self.__downcast(df)
        
        return df

    def __getUniqueShopItems(self, train):
    
        unique_shop_items = train[['shop_id','item_id']].drop_duplicates(keep='first')
        x = pd.DataFrame(unique_shop_items.groupby(['item_id']).agg({'shop_id':'count'}).rename(columns={'shop_id':'is_unique_to_shop_item'})).reset_index()
        x = x[(x.is_unique_to_shop_item == 1)]
        unique_shop_items = unique_shop_items.set_index(['item_id'])
        unique_shop_items = unique_shop_items.merge(x, left_index=True, right_on=['item_id'], how='left').fillna(0)
        unique_shop_items.is_unique_to_shop_item = np.int8(unique_shop_items.is_unique_to_shop_item)
        unique_shop_items.shop_id = np.int32(unique_shop_items.shop_id)
        unique_shop_items.item_id = np.int32(unique_shop_items.item_id)
        
        return unique_shop_items
    

    def __aggregateTrainByMonth(self, train, items, categories, verbose=False):
        #clean and aggregate training data
        
        if verbose:
            print("\tCleaning known training errors and duplicates...")
        ### CLEAN KNOWN ISSUES ###
        #a negative price exists in the train set; fix it first before aggregation
        train.loc[(train.item_price == -1), 'item_price'] = 1249.0

        #there is also a 'giant' item price out there; a singular value of 307,960
        #it is currently a product called Radmin 3 and item 6066 is a multiple of another product, 6065 (522x vs. 1x)
        #item 6066 never happens in the test set, but the train set has it.  Let's get rid of 6066 and change it with 6065
        train.loc[(train.item_id == 6066), ['item_price','item_id','item_cnt_day']] =  [1299.0, 6065, 1]

        #drop duplicates
        train.date = pd.to_datetime(train.date, format='%d.%m.%Y', errors='coerce')
        train = train.drop_duplicates(keep='first') 

        if verbose:
            print("\tAggregating item quantities on a per item + shop + date block basis...")
        
        train = pd.DataFrame(train.groupby(['item_id','shop_id','date_block_num'], 
            as_index=False).agg({'item_cnt_day':['sum','count','std']})).fillna(0.0)
        train.columns = ['item_id','shop_id','date_block_num','item_cnt_month','customer_sales','item_cnt_month_std']
        
        train.item_cnt_month = np.clip(train.item_cnt_month, 0, 20)
        train.loc[(train.item_cnt_month == 0) & (train.customer_sales > 0), 'customer_sales'] = 0
        
        if verbose:
            print("\tCreating cartesian product of all shops and items on a per date block basis to simulate TEST dataset...")
        #create cartesian products of shops x items on a per month basis
        train_temp = []
        for i in range(train.date_block_num.min(), (train.date_block_num.max() + 1), 1):
            date_slice = train[(train.date_block_num == i)]
            train_temp.append(np.array(cartesian((date_slice.item_id.unique(), date_slice.shop_id.unique(), [i]))))

        
        train_temp = pd.DataFrame(np.vstack(train_temp), columns = ['item_id','shop_id', 'date_block_num'], dtype=np.int32)
        train = pd.merge(train_temp, train, on=['item_id','shop_id','date_block_num'], how='left').fillna(0.0)

        if verbose:
            print("\tcalculating shop-only and item-only total sales by month...")
        shoponly_sales = pd.DataFrame(train.groupby(['date_block_num','shop_id'])['item_cnt_month'].transform('sum')).rename(columns={'item_cnt_month':'item_cnt_month_shoponly'})
        itemonly_sales = pd.DataFrame(train.groupby(['date_block_num','item_id'])['item_cnt_month'].transform('sum')).rename(columns={'item_cnt_month':'item_cnt_month_itemonly'})
        
        train = train.merge(shoponly_sales, left_index=True, right_index=True, how='inner')
        train = train.merge(itemonly_sales, left_index=True, right_index=True, how='inner')

        if verbose:
            print("\tcalculating shop-only and item-only STD sales by month...")
        shoponly_sales = pd.DataFrame(train.groupby(['date_block_num','shop_id'])['item_cnt_month'].transform('std')).rename(columns={'item_cnt_month':'item_cnt_month_std_shoponly'})
        itemonly_sales = pd.DataFrame(train.groupby(['date_block_num','item_id'])['item_cnt_month'].transform('std')).rename(columns={'item_cnt_month':'item_cnt_month_std_itemonly'})
        
        train = train.merge(shoponly_sales, left_index=True, right_index=True, how='inner')
        train = train.merge(itemonly_sales, left_index=True, right_index=True, how='inner')

        del shoponly_sales, itemonly_sales
        gc.collect()

        if verbose:
            print("\tAdding item categories and super categories...")
        #add item category and item super category to train
        train = train.merge(items[['item_id','item_category_id']], on=['item_id'], how='left')
        train = train.merge(categories[['item_category_id','super_category_id']], on=['item_category_id'], how='left')

        if verbose:
            print("\tCalculating cateogory and super-category quantities per month...")
        #get prices by shop + item + month + category
        avg_qty = pd.DataFrame(train.groupby(['item_category_id','shop_id','date_block_num'], 
            as_index=False).agg({'item_cnt_month':'sum'})).fillna(0.0)
        avg_qty.columns = ['item_category_id','shop_id','date_block_num','item_category_cnt_month']
        
        train = train.merge(avg_qty, on=['item_category_id','shop_id','date_block_num'], how='left').fillna(0.0)
        
        avg_qty = pd.DataFrame(train.groupby(['super_category_id','shop_id','date_block_num'], 
            as_index=False).agg({'item_cnt_month':'sum'})).fillna(0.0)
        avg_qty.columns = ['super_category_id','shop_id','date_block_num','super_category_cnt_month']
        
        train = train.merge(avg_qty, on=['super_category_id','shop_id','date_block_num'], how='left').fillna(0.0)

        if verbose:
            print("\tCleaning up temporary objects...")
        del avg_qty, train_temp
        gc.collect()
        
        train = self.__downcast(train)
        return train
    
    def __orderTrainTest(self, train, test, cols):
        train = train[cols]
        test = test[cols]
        return train, test

      

    def __populateTestItemCategories(self, test, items, categories, verbose=False):
        #populate item prices for the test set based on previous item + shop combos in the dataset
        #additionally, add item category to the end
        
        if verbose:
            print("\tadding category and super category to TEST data...")
        #add item category and item super category to train
        test = test.merge(items[['item_id','item_category_id']], on=['item_id'], how='left')
        test = test.merge(categories[['item_category_id','super_category_id']], on=['item_category_id'], how='left')
        
        test = test.set_index('tuple_id')
        test['date_block_num'] = np.int32(34)
        test = self.__downcast(test)
        
        return test
    
    def __massParallelPeriodShift (self, train, test, pp_range = [1,2,3,6,12,24], shift_cols = ['item_cnt_month'], encode=False, encode_type='mean', verbose=False, clipping=False):
        #iterate through a mass list of columns to get parallel period shift values
        z_iter = np.array(list(product(*[pp_range, shift_cols])))
        
        drop_labs = list(train.columns.difference(test.columns))
        test = test.reset_index()
        drop_labs.append('is_train')
        train['is_train'] = np.int8(1)
        test['is_train'] = np.int8(0)
        
        mrg = pd.concat([train,test],axis=0)
        
        for val in z_iter:
            if verbose:
                print("\tgetting shifted [%d] parallel period for '%s'..." % (int(val[0]), val[1]))
            mrg, newcolname = self.__parallelPeriodShiftBy(mrg, int(val[0]), val[1])
            if encode:
                encode_col = encode_type.upper() + '_' + newcolname

                if verbose:
                    print("\t%s encoding new column as '%s'..." % (encode_type, encode_col))
                mrg = self.__encodeBoth(mrg, ['shop_id','item_id'], newcolname, encode_col, encode_type=encode_type)
            if clipping:
                mrg[newcolname] = np.clip(mrg[newcolname], 0, 20)
                mrg[encode_col] = np.clip(mrg[encode_col], 0, 20)
        
        train = mrg[(mrg.is_train == 1)]
        test = mrg[(mrg.is_train == 0)]
        
        train = train.drop(labels=['is_train','tuple_id'], axis=1)
        test = test.set_index('tuple_id')
        
        test = test.drop(labels=drop_labs, axis=1)

        del mrg
        gc.collect()
        
        return train, test
        
    
    def __parallelPeriodShiftBy (self, df, pp_num = 1, shift_col = 'item_cnt_month'):
        pps_col = 'pps_{}_lag_{}'.format(shift_col, pp_num)
        index_cols = ['shop_id','item_id','date_block_num']
        get_cols = list(index_cols)
        get_cols.append(shift_col)
        set_cols = list(index_cols)
        set_cols.append(pps_col)
        
        shifted_train = df.loc[(df.item_cnt_month > 0), get_cols].copy()
        shifted_train['date_block_num'] += pp_num
        shifted_train.columns=set_cols
        
        #test = test.merge(shifted_train, on=index_cols, how='left').fillna(0.0)
        #train = train.merge(shifted_train, on=index_cols, how='left').fillna(0.0)
        df = df.merge(shifted_train, on=index_cols, how='left').fillna(0.0)
        
        return df, pps_col
    
    def __parallelPeriodOccurrence (self, train, test, pp_num = 1, shifted_col = 'item_cnt_month', verbose=False, mean_encode=False, clipping=False):
        ppo_col = 'ppo_{}_lag_{}'.format(shifted_col, pp_num)
        index_cols = ['shop_id','item_id','date_block_num']
        get_cols = list(index_cols)
        get_cols.append(shifted_col)
        set_cols = list(index_cols)
        set_cols.append(ppo_col)
        
        if verbose:
            print("\tpopulating prior occurrence [%d] of '%s' column for TEST data (saved as '%s'..." % (pp_num, shifted_col, ppo_col))
        #retrieve the maximum date_block_num for all shop + item combos where a real item entry exists (not zero / backfilled)
        shifted_train = pd.DataFrame(train.loc[(train.item_cnt_month > 0)][index_cols].groupby(['shop_id','item_id'], as_index=True).agg({'date_block_num':'max'})).reset_index()
        shifted_train = shifted_train.merge(train[(train.item_cnt_month > 0)][get_cols], on=index_cols, how='inner')
        shifted_train.columns = set_cols
        shifted_train.date_block_num = 34  #match the TEST data set
        if clipping:
            shifted_train[ppo_col] = np.clip(shifted_train[ppo_col], 0, 20)


        test = test.merge(shifted_train, on=index_cols, how='left').set_index(test.index).fillna(0.0)

        train_train = pd.DataFrame(columns=set_cols)
        train_train.shop_id = shifted_train.shop_id.astype(np.int32)
        train_train.item_id = shifted_train.item_id.astype(np.int32)
        train_train.date_block_num = shifted_train.date_block_num.astype(np.int32)
        train_train[ppo_col] = shifted_train[ppo_col].astype(np.float32)
        
        if verbose:
            print("\tpopulating prior occurrence [%d] of '%s' column for TRAIN data 1 period at a time (saved as '%s'..." % (pp_num, shifted_col, ppo_col))
        for i in range(train.date_block_num.min() + 1, (train.date_block_num.max() + 1), 1):
            if verbose:
                print("\t\tgenerating data for data block [%d] of [%d]..." % (i, train.date_block_num.max()))
            shifted_train = pd.DataFrame(train.loc[(train.item_cnt_month > 0) & (train.date_block_num < i)][index_cols].groupby(['shop_id','item_id'], as_index=True).agg({'date_block_num':'max'})).reset_index()
            shifted_train = shifted_train.merge(train[(train.item_cnt_month > 0) & (train.date_block_num < i)][get_cols], on=index_cols, how='inner')
            shifted_train.columns = set_cols
            shifted_train.date_block_num = i #update to the current loop value to alow for a join
            if clipping:
                shifted_train[ppo_col] = np.clip(shifted_train[ppo_col], 0, 20)
                
            train_train = pd.concat([train_train, shifted_train], axis=0, ignore_index=True)
        
        if verbose:
            print("\t\tcombining the TRAIN data prior occurrences...")
        train = train.merge(train_train, on=index_cols, how='left').fillna(0.0)
        
        
        if mean_encode:
            if verbose:
                print("\t mean enconding '%s' column for TEST and TRAIN..." % ppo_col)
            
            train['is_train'] = np.int8(1)
            test['is_train'] = np.int8(0)
            drop_labs = list(train.columns.difference(test.columns))
            drop_labs.append('is_train')

            
            test = test.reset_index()
     
            mrg = pd.concat([train,test],axis=0)
            
            
            encode_col = 'MEAN_' + ppo_col


            train = mrg[(mrg.is_train == 1)]
            test = mrg[(mrg.is_train == 0)]

            mrg = self.__encodeBoth(mrg, ['shop_id','item_id'], ppo_col, encode_col, encode_type='mean')
            #train = self.__encode(train, train, ['shop_id','item_id'], ppo_col, encode_col, encode_type='mean')
            if clipping:
                mrg[ppo_col] = np.clip(mrg[ppo_col], 0, 20)
            
            train = mrg[(mrg.is_train == 1)]
            test = mrg[(mrg.is_train == 0)]
    
            train = train.drop(labels=['is_train','tuple_id'], axis=1)
            test = test.set_index('tuple_id')
            
            test = test.drop(labels=drop_labs, axis=1)

            del mrg
            gc.collect()

        del shifted_train, train_train
        gc.collect()
        
        return train, test
   
    def __parallelPeriodBy (self, train, test, col_append, pp_num=1, period_col='date_block_num', group_by_cols=['shop_id','item_id'], agg_method='mean'):
        colpart = list(group_by_cols)
        colpart = '_'.join(colpart)
        col_cnt = 'pp_' + agg_method.upper() + '_' + colpart + '_period_' + str(pp_num) + '_by_' + period_col
        
        col_append.append(col_cnt)
        
        group_by_cols.append(period_col)
        group_1 = list(group_by_cols)
        group_1.append('item_cnt_month')
        group_2 = list(group_by_cols)
        group_2.append(col_cnt)
        
        shifted_train = train[group_1].drop_duplicates(keep='first')
        
        shifted_train = pd.DataFrame(train[group_1].groupby(group_by_cols, as_index=False).agg({'item_cnt_month':agg_method}))
        shifted_train.columns = group_2  #this step renames 'item_cnt_month' to the new name that will be appended
        shifted_train[period_col] += pp_num  #shift the period number forward by the lag period so join operation lines up


        train = train.merge(shifted_train, on=group_by_cols, how='left')
        test = test.merge(shifted_train, on=group_by_cols, how='left')

        train[col_cnt].fillna(0, inplace=True)
        test[col_cnt].fillna(0, inplace=True)
        
        return train, test, col_append 
    
    def __encode (self, df, train, col_groupBy_arr, col_target, col_newName, include_zeroes=False, encode_type='mean'):
        if include_zeroes:
            mean_val = train[col_target].mean()
        else:
            mean_val = train[(df.item_price_max > 0)][col_target].mean()
        
        df[col_newName] = df.groupby(col_groupBy_arr)[col_target].transform(encode_type)
        df[col_newName].fillna(mean_val, inplace=True)
        
        return df

    def __encodeBoth (self, df, col_groupBy_arr, col_target, col_newName, encode_type='mean'):

        mean_val = df[col_target].mean()
        
        df[col_newName] = df[(df[col_target] != 0)].groupby(col_groupBy_arr)[col_target].transform(encode_type)#.fillna(mean_val)
        df[col_newName].fillna(mean_val, inplace=True)
        
        return df
    
    def __periodization (self, df):
        #creates the quarter, half, year
        
        quarterMap = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2,
                      9:3, 10:3, 11:3, 12:4, 13:4, 14:4, 15:5, 16:5, 17:5,
                      18:6, 19:6, 20:6, 21:7, 22:7, 23:7, 24:8, 25:8, 26:8,
                      27:9, 28:9, 29:9, 30:10, 31:10, 32:10, 33:11, 34:11}

        halfMap = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2,
                   6:3, 7:3, 8:4, 9:4, 10:5, 11:5}

        df['quarter'] = df.date_block_num.map(quarterMap)
        df.quarter = pd.to_numeric(df.quarter, errors='coerce')
        
        df['half'] = df.quarter.map(halfMap)
        df.half = pd.to_numeric(df.half, errors='coerce')
        
        df.loc[(df.date_block_num <= 11), 'year'] = 2013
        df.loc[((df.date_block_num >11) & (df.date_block_num <= 23)), 'year'] = 2014
        df.year = df.year.fillna(2015)
        df.year = pd.to_numeric(df.year, errors='coerce')
        
        return df
    
    def __refactorItemSales(self, train):
       
        dbn = list(range(0,24,1))
        fact = [0.768520271, 0.64928214, 0.565586266, 0.60631028, 0.606213764, 0.536732014,
             0.554263349, 0.539800973, 0.528098016, 0.574072003, 0.595210838, 0.595210838,
             0.90093114, 0.784335296, 0.746631278, 0.737461388, 0.685120203, 0.662168352,
             0.70471632, 0.655380177, 0.697692987, 0.66700965, 0.728532071, 0.728532071]
        factors = pd.DataFrame(fact, dbn, columns=['factor'])
        factors.index.name = 'date_block_num'        

        train = train.merge(factors, left_on=['date_block_num'], right_index=True, how='left').fillna(1.0)
        train.item_cnt_month = train.item_cnt_month * train.factor
        train = train.drop(labels=['factor'], axis=1)
 
        return train

    def __priorPeriodRate(self, train, test, col_append):
        #NOTE: This routine can not be used in conjuction with refactorItemSales routine...
        #perform aggregation at the macro 'date block' only level.... could do Item + shop for more specific tuning, but hard to carry forward in test
        roc = train.groupby(['date_block_num'], as_index=False).agg({'item_cnt_month':'sum'})
        roc.columns=['date_block_num','cnt_sum']
        
        upd = roc.copy()
        upd.date_block_num += 1
        upd.columns=['date_block_num','pp_cnt_sum']
        roc = roc.merge(upd, on=['date_block_num'], how='left')
        roc['pp_rate'] = roc.pp_cnt_sum / roc.cnt_sum
        roc.pp_rate = roc.pp_rate.fillna(1.0)
        
        train = train.merge(roc[['date_block_num','pp_rate']], on=['date_block_num'], how='left')
        train.pp_rate = train.pp_rate.fillna(1.0)
        test['pp_rate'] = 1.739116314567332
        
        col_append.append('pp_rate')
        return train, test, col_append   
    
    def __cantorPairingFunction(self, train, test):
        #create a cantor pairing for item + test
        
        train['cantor'] = (0.5 * (train.item_category_id + train.shop_id) * (train.item_category_id + train.shop_id + 1) + train.shop_id)
        test['cantor'] = (0.5 * (test.item_category_id + test.shop_id) * (test.item_category_id + test.shop_id + 1) + test.shop_id)
        
        return train, test
    
    def __createInteraction(self, df, col1, col2, method_sum = True, method_mult = True, method_diff = True, method_divide = True, method_mean = True):
    
        cname= 'INTERACT__' + col1 + '__' + col2
        
        if method_sum:
            df[cname + '_SUM'] = df[col1] + df[col2]
        if method_mult:
            df[cname + '_MULT'] = df[col1] * df[col2]
        if method_diff:
            df[cname + '_DIFF'] = df[col1] - df[col2]
        if method_divide:
            df[cname + '_DIV'] = df[col1].divide(df[col2]).replace([np.inf, -np.inf, np.nan], 0.0)
            #df[cname + '_DIV'] = df[cname + '_DIV'].fillna(0.0)
        if method_mean:
            df[cname + '_MEAN'] = np.mean((df[col1], df[col2]))

        return df

    def __expandingMean(self, train, test, group_cols = ['shop_id','item_id'], col_to_mean = 'MEAN_pps_item_cnt_month_lag_1', create_interaction=True):
        #note : currently, all interactions are division only for expanding mean
        
        expcol = 'exp_mean_' + col_to_mean + '_by_' + '_'.join(group_cols)
        #exp_mean_MEAN_pps_item_cnt_month_lag_1_by_shop_id_item_id  is default value
        
        drop_labs = list(train.columns.difference(test.columns))
        test = test.reset_index()
        drop_labs.append('is_train')
        train['is_train'] = np.int8(1)
        test['is_train'] = np.int8(0)
                
        mrg = pd.concat([train,test],axis=0)
        
        sort_cols = ['date_block_num']
        sort_cols.extend(list(group_cols))
        
        mrg = mrg.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        mrg[expcol] = mrg.groupby(group_cols)[col_to_mean].transform(lambda x: x.expanding().mean())
        
        if create_interaction:
            mrg = self.__createInteraction(mrg, col_to_mean, expcol, method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        train = mrg[(mrg.is_train == 1)]
        test = mrg[(mrg.is_train == 0)]
                
        train = train.drop(labels=['is_train','tuple_id'], axis=1)
        test = test.sort_values(by=['tuple_id'], ascending=True)
        test = test.set_index('tuple_id')
                
        test = test.drop(labels=drop_labs, axis=1)
       
        return train, test
    
    def __rollingWindow(self, train, test, group_cols = ['shop_id','item_id'], window_size = 2, col_to_roll = 'pps_item_cnt_month_lag_1', create_interaction=True):
        rcol = 'rw' + str(window_size) + '_' + col_to_roll + '_by_' + '_'.join(group_cols)
        #rw2_pps_item_cnt_month_lag_1_by_shop_id_item_id  is default value
        
        drop_labs = list(train.columns.difference(test.columns))
        test = test.reset_index()
        drop_labs.append('is_train')
        train['is_train'] = np.int8(1)
        test['is_train'] = np.int8(0)
                
        mrg = pd.concat([train,test],axis=0)
        
        sort_cols = ['date_block_num']
        sort_cols.extend(list(group_cols))
        
        mrg = mrg.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        mrg[rcol] = mrg.groupby(group_cols)[col_to_roll].transform(lambda x: x.rolling(window=window_size).mean())
        mrg[rcol] = mrg[rcol].where(mrg[rcol].notnull(), mrg[col_to_roll])
        
        if create_interaction:
            mrg = self.__createInteraction(mrg, col_to_roll, rcol, method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        train = mrg[(mrg.is_train == 1)]
        test = mrg[(mrg.is_train == 0)]
                
        train = train.drop(labels=['is_train','tuple_id'], axis=1)
        test = test.sort_values(by=['tuple_id'], ascending=True)
        test = test.set_index('tuple_id')
                
        test = test.drop(labels=drop_labs, axis=1)
       
        return train, test

    def __ewma(self, train, test, group_cols = ['shop_id','item_id'], alpha = 0.5, col_to_mean = 'pps_item_cnt_month_lag_1', create_interaction=True):
        ewmacol = 'ewma_' + col_to_mean + '_by_' + '_'.join(group_cols)
        #ewma_pps_item_cnt_month_lag_1_by_shop_id_item_id  is default value
        
        drop_labs = list(train.columns.difference(test.columns))
        test = test.reset_index()
        drop_labs.append('is_train')
        train['is_train'] = np.int8(1)
        test['is_train'] = np.int8(0)
                
        mrg = pd.concat([train,test],axis=0)
        
        sort_cols = ['date_block_num']
        sort_cols.extend(list(group_cols))
        
        mrg = mrg.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        mrg[ewmacol] = mrg.groupby(group_cols)[col_to_mean].transform(lambda x: x.ewm(alpha=alpha, min_periods=0).mean())
        mrg[ewmacol] = mrg[ewmacol].where(mrg[ewmacol].notnull(), mrg[col_to_mean])
        
        if create_interaction:
            mrg = self.__createInteraction(mrg, col_to_mean, ewmacol, method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        train = mrg[(mrg.is_train == 1)]
        test = mrg[(mrg.is_train == 0)]
                
        train = train.drop(labels=['is_train','tuple_id'], axis=1)
        test = test.sort_values(by=['tuple_id'], ascending=True)
        test = test.set_index('tuple_id')
                
        test = test.drop(labels=drop_labs, axis=1)
       
        return train, test    
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    #### SUBMISSION FILE GENERATION 
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    
    def GenerateSubmissionFiles(self, verbose=True):
        if verbose:
            print("\nbuilding submission train / test sets:\n--------------------------------------\n")

        #open train and item files only
        if verbose:
            print("opening original train, test, and items files...")
        train = self.__openFilePrepared(self.__orig_trainDataFile)
        test = self.__openFilePrepared(self.__orig_testDataFile)
        items = self.__openFilePrepared(self.__orig_itemDataFile)
        categories = self.__openFilePrepared(self.__orig_categoryDataFile)

        
        #unique_shop_items = self.__getUniqueShopItems(train)
        
        #aggregate train data to month+year (date_block_num) grain
        if verbose:
            print("Preparing TRAIN data set:")
        #new_items = np.setdiff1d(np.sort(test.item_id.unique()), np.sort(train.item_id.unique()))
        train = self.__aggregateTrainByMonth(train, items, categories, verbose)
        
        #arrange test with starting features
        if verbose:
            print("Preparing TEST data set:")
        test.columns = ['tuple_id','shop_id','item_id']
        #populate the item prices based on the LAST price the shop sold the item form
        test = self.__populateTestItemCategories(test, items, categories, verbose)

        del items, categories
        gc.collect()

        if verbose:
            print("Generating shifted parallel period features:")
        train, test = self.__massParallelPeriodShift (train, test, 
            pp_range = [1,2,3,6,12], shift_cols = list(train.columns.difference(test.columns)), 
            encode=True, encode_type='mean', verbose=verbose, clipping=False)

        if verbose:
            print("Generating rolling mean averages...")
        train, test = self.__rollingWindow(train, test)

        if verbose:
            print("Generating exponentially weighted mean averages...")
        train, test = self.__ewma(train, test)
   
        if verbose:
            print("Creating a pps cumsum_exp_mean_shop_id_item_id feature...")
        
        cumsum = train.groupby(['shop_id','item_id'])['item_cnt_month'].cumsum() - train.item_cnt_month
        cumcnt = train.groupby(['shop_id','item_id']).cumcount()
        train['cumsum_exp_mean_shop_id_item_id'] = cumsum / cumcnt
        #mean_val = train['cumsum_exp_mean_shop_id_item_id'].mean()  # this one includes all the zero's
        mean_val = np.float32(0.0)
        print("\tmissing values in cumsum_exp_mean_shop_id_item_id (imputing to %f): %d" % (mean_val, train.cumsum_exp_mean_shop_id_item_id.isnull().sum()))
        train.cumsum_exp_mean_shop_id_item_id = train.cumsum_exp_mean_shop_id_item_id#.fillna(0.0)
        train.cumsum_exp_mean_shop_id_item_id.fillna(mean_val, inplace=True)
        train, test = self.__massParallelPeriodShift (train, test, 
            pp_range = [1], shift_cols = ['cumsum_exp_mean_shop_id_item_id'], 
            encode=True, encode_type='mean', verbose=verbose, clipping=False)
        train = train.drop(labels=['cumsum_exp_mean_shop_id_item_id'], axis=1)
     
        
        if verbose:
            print("Generating prior period occurrence for -1 prior period:")
        #train, test = self.__parallelPeriodOccurrence (train, test, pp_num = 1, shifted_col = 'item_cnt_month', verbose=verbose, mean_encode=True, clipping=False)
        train, test = self.__parallelPeriodOccurrence (train, test, pp_num = 1, shifted_col = 'item_category_cnt_month', verbose=verbose, mean_encode=True, clipping=False)
        
        if verbose:
            print("Creating interaction columns...")
        #diff column for item price
        #train = self.__createInteraction(train, 'item_category_price_max', 'item_category_price_min', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        #test = self.__createInteraction(test, 'item_category_price_max', 'item_category_price_min', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        #train = self.__createInteraction(train, 'super_category_price_max', 'super_category_price_min', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        #test = self.__createInteraction(test, 'super_category_price_max', 'super_category_price_min', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        #train = self.__createInteraction(train, 'item_price_max', 'item_price_min', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        #test = self.__createInteraction(test, 'item_price_max', 'item_price_min', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_12', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_12', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_6', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_6', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_3', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_3', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        #train = self.__createInteraction(train, 'MEAN_ppo_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=True)
        #test = self.__createInteraction(test, 'MEAN_ppo_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=True)

        #train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'item_price_min', method_sum=False, method_mult=True, method_diff=False, method_divide=False, method_mean=False)
        #test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'item_price_min', method_sum=False, method_mult=True, method_diff=False, method_divide=False, method_mean=False)

        #train = self.__createInteraction(train, 'INTERACT__MEAN_pps_item_cnt_month_lag_1__MEAN_pps_item_cnt_month_lag_12_DIFF', 'item_price_min', method_sum=False, method_mult=True, method_diff=False, method_divide=False, method_mean=False)
        #test = self.__createInteraction(test, 'INTERACT__MEAN_pps_item_cnt_month_lag_1__MEAN_pps_item_cnt_month_lag_12_DIFF', 'item_price_min', method_sum=False, method_mult=True, method_diff=False, method_divide=False, method_mean=False)

        train = self.__createInteraction(train, 'pps_item_cnt_month_lag_1', 'pps_item_cnt_month_shoponly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        test = self.__createInteraction(test, 'pps_item_cnt_month_lag_1', 'pps_item_cnt_month_shoponly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        
        train = self.__createInteraction(train, 'pps_item_cnt_month_lag_1', 'pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        test = self.__createInteraction(test, 'pps_item_cnt_month_lag_1', 'pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        #train = self.__createInteraction(train, 'ppo_item_cnt_month_lag_1', 'pps_item_cnt_month_shoponly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        #test = self.__createInteraction(test, 'ppo_item_cnt_month_lag_1', 'pps_item_cnt_month_shoponly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        
        #train = self.__createInteraction(train, 'ppo_item_cnt_month_lag_1', 'pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        #test = self.__createInteraction(test, 'ppo_item_cnt_month_lag_1', 'pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        #train = self.__createInteraction(train, 'item_price_max', 'pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        #test = self.__createInteraction(test, 'item_price_max', 'pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        train = self.__createInteraction(train, 'pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        test = self.__createInteraction(test, 'pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        train = self.__createInteraction(train, 'pps_item_cnt_month_itemonly_lag_1', 'MEAN_pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        test = self.__createInteraction(test, 'pps_item_cnt_month_itemonly_lag_1', 'MEAN_pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        #train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=True, method_mean=False)
        #test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'pps_item_cnt_month_itemonly_lag_1', method_sum=False, method_mult=False, method_diff=True, method_divide=True, method_mean=False)
        #train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_2', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)
        #test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'MEAN_pps_item_cnt_month_lag_2', method_sum=False, method_mult=False, method_diff=True, method_divide=False, method_mean=False)

        #train, test = self.__cantorPairingFunction(train, test)

        train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'pps_item_cnt_month_lag_1', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        train = self.__createInteraction(train, 'MEAN_pps_item_cnt_month_lag_1', 'ewma_pps_item_cnt_month_lag_1_by_shop_id_item_id', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)
        test = self.__createInteraction(test, 'MEAN_pps_item_cnt_month_lag_1', 'ewma_pps_item_cnt_month_lag_1_by_shop_id_item_id', method_sum=False, method_mult=False, method_diff=False, method_divide=True, method_mean=False)

        #limit train to 2015 only at this point
        ###train = train[(train.date_block_num >= 24)]

        #if verbose:
        #    print("Generating expanding mean for shop + item on MEAN_pps_item_cnt_month_lag_1...")
        #    train, test = self.__expandingMean (train, test, ['shop_id','item_id'], 'MEAN_pps_item_cnt_month_lag_1', create_interaction=True)

        if verbose:
            print("Generating expanding mean for shop + item on pps_item_cnt_month_lag_1...")
            train, test = self.__expandingMean (train, test, ['shop_id','item_id'], 'pps_item_cnt_month_lag_1', create_interaction=True)

        if verbose:
            print("Generating expanding mean for item on pps_item_cnt_month_itemonly_lag_1...")
            train, test = self.__expandingMean (train, test, ['item_id'], 'pps_item_cnt_month_itemonly_lag_1', create_interaction=True)

        if verbose:
            print("Generating expanding mean for shop on pps_item_cnt_month_shoponly_lag_1...")
            train, test = self.__expandingMean (train, test, ['shop_id'], 'pps_item_cnt_month_shoponly_lag_1', create_interaction=True)

        if verbose:
                print("adding a rolling mean for pps item count for 2 and 3 periods...")
        train['AAAAA_ravg_prior2'] = (train.pps_item_cnt_month_lag_1 + train.pps_item_cnt_month_lag_2) / 2
        train['AAAAA_ravg_prior3'] = (train.pps_item_cnt_month_lag_1 + train.pps_item_cnt_month_lag_2 + train.pps_item_cnt_month_lag_3) / 3
        test['AAAAA_ravg_prior2'] = (test.pps_item_cnt_month_lag_1 + test.pps_item_cnt_month_lag_2) / 2
        test['AAAAA_ravg_prior3'] = (test.pps_item_cnt_month_lag_1 + test.pps_item_cnt_month_lag_2 + test.pps_item_cnt_month_lag_3) / 3


        test = test.reset_index()
        test.tuple_id = np.int32(test.tuple_id)
        test = test.set_index('tuple_id')
        test.index.name = 'ID'
        
        train.index.name = 'ID'

        '''
        if verbose:
            print("Classifying row based on apperaance of shop + item combo in the last 1, 2, and 3 months...")
        train['present_last_month'] = 1 * train.pps_item_cnt_month_lag_1
        train['present_last_two_months'] = train['present_last_month'] * train.pps_item_cnt_month_lag_2
        train['present_last_three_months'] = train['present_last_two_months'] * train.pps_item_cnt_month_lag_3
        train['present_last_month'] = np.int8(np.clip(train['present_last_month'], 0, 1))
        train['present_last_two_months'] = np.int8(np.clip(train['present_last_two_months'], 0, 1))
        train['present_last_three_months'] = np.int8(np.clip(train['present_last_three_months'], 0, 1))
        
        test['present_last_month'] = 1 * test.pps_item_cnt_month_lag_1
        test['present_last_two_months'] = test['present_last_month'] * test.pps_item_cnt_month_lag_2
        test['present_last_three_months'] = test['present_last_two_months'] * test.pps_item_cnt_month_lag_3
        test['present_last_month'] = np.int8(np.clip(test['present_last_month'], 0, 1))
        test['present_last_two_months'] = np.int8(np.clip(test['present_last_two_months'], 0, 1))
        test['present_last_three_months'] = np.int8(np.clip(test['present_last_three_months'], 0, 1))
        '''
        
        #train = train.merge(unique_shop_items, on=['shop_id','item_id'], how='left').fillna(0)
        #test = test.merge(unique_shop_items, on=['shop_id','item_id'], how='left').fillna(0)
        
        #train.is_unique_to_shop_item = np.int8(train.is_unique_to_shop_item)
        #test.is_unique_to_shop_item = np.int8(test.is_unique_to_shop_item)
        
        if verbose:
            print("INFO: shape of TEST:", test.shape)
            print("INFO: shape of TRAIN:", train.shape)

        #prepare dataframes for export to CSV
        if verbose:
            print("Splitting train dataframe into data and labels for output...")
        
        #dates = train['date_block_num']
        
        train_labels = train[['item_cnt_month']]
        #train_labels.item_cnt_month = np.clip(train_labels.item_cnt_month, 0, 20)
        
        
        train = train.drop(labels=['item_cnt_month', 'item_cnt_month_itemonly', 'item_cnt_month_shoponly', 
                'item_category_cnt_month', 'super_category_cnt_month', 'customer_sales', 'item_cnt_month_std', 'item_cnt_month_std_shoponly', 'item_cnt_month_std_itemonly'], axis=1)

        restrict = [
        'item_cnt_month', 'item_cnt_month_itemonly', 'item_cnt_month_shoponly', 
        'item_category_cnt_month', 'super_category_cnt_month', 'customer_sales',
        'rw2_pps_item_cnt_month_lag_1_by_shop_id_item_id',
        'super_category_id',
        'pps_item_category_cnt_month_lag_1',
        'MEAN_pps_item_category_cnt_month_lag_12',
        'pps_super_category_cnt_month_lag_12',
        'MEAN_pps_super_category_cnt_month_lag_6',
        'pps_item_cnt_month_lag_12',
        'MEAN_pps_super_category_cnt_month_lag_12',
        'pps_item_cnt_month_shoponly_lag_6',
        'MEAN_pps_item_cnt_month_shoponly_lag_6',
        'pps_item_cnt_month_shoponly_lag_12',
        'MEAN_pps_item_cnt_month_shoponly_lag_12',
        'pps_item_cnt_month_lag_1',
        'MEAN_pps_item_cnt_month_lag_12',
        'pps_super_category_cnt_month_lag_6',
        'pps_item_category_cnt_month_lag_6',
        'pps_item_category_cnt_month_lag_12',
        'pps_item_cnt_month_lag_6',
        'MEAN_pps_item_category_cnt_month_lag_3',
        'MEAN_pps_item_category_cnt_month_lag_2',
        'MEAN_pps_item_category_cnt_month_lag_1',
        'MEAN_pps_item_cnt_month_itemonly_lag_12',
        'MEAN_pps_item_cnt_month_itemonly_lag_6',
        'MEAN_pps_item_category_cnt_month_lag_6',
        'pps_super_category_cnt_month_lag_2',
        'MEAN_pps_super_category_cnt_month_lag_3',
        'MEAN_pps_item_cnt_month_shoponly_lag_3',
        'pps_item_cnt_month_lag_3',
        'pps_item_cnt_month_shoponly_lag_2',
        'pps_item_cnt_month_lag_2',
        'MEAN_pps_super_category_cnt_month_lag_2',
        'pps_item_cnt_month_itemonly_lag_12',
        'pps_item_category_cnt_month_lag_2',
        'pps_item_category_cnt_month_lag_3',
        'pps_item_cnt_month_shoponly_lag_3',
        'pps_item_cnt_month_shoponly_lag_1',
        'pps_super_category_cnt_month_lag_3',
        'MEAN_pps_item_cnt_month_lag_6',
        'MEAN_pps_item_cnt_month_lag_3',
        'INTERACT__pps_item_cnt_month_lag_1__rw2_pps_item_cnt_month_lag_1_by_shop_id_item_id_DIV',
        'pps_item_cnt_month_itemonly_lag_6',
        'MEAN_pps_super_category_cnt_month_lag_1',
        'INTERACT__pps_item_cnt_month_lag_1__exp_mean_pps_item_cnt_month_lag_1_by_shop_id_item_id_DIV',
        'MEAN_pps_item_cnt_month_shoponly_lag_2',
        'MEAN_pps_item_cnt_month_itemonly_lag_3',
        'pps_super_category_cnt_month_lag_1',
        'MEAN_pps_item_cnt_month_itemonly_lag_2', 'super_category_cnt_month']

        #predictors = np.setdiff1d(np.sort(train.columns), np.sort(restrict))
        train = train[np.setdiff1d(np.sort(train.columns), np.sort(restrict))]
        test = test[np.setdiff1d(np.sort(test.columns), np.sort(restrict))]
        
        
        if verbose:
            print("Ordering columns in train and test to match...")
        train, test = self.__orderTrainTest(train, test, train.columns)

        #product final output files
        if verbose:
            print("\nProducing output files now:\n---------------------------")
        test.to_csv(self.__out_testDataFile)
        if verbose:
            print("submission test data file [%s] generated." % (self.__out_testDataFile))

        train.to_csv(self.__out_trainDataFile)
        if verbose:
            print("submission train data file [%s] generated." % (self.__out_trainDataFile))
        train_labels.to_csv(self.__out_trainLabelsFile)
        if verbose:
            print("submission train labels file [%s] generated." % (self.__out_trainLabelsFile))

        '''
        train.to_csv(self.__out_trainDataFile.replace('.csv', '_2015.csv'))
        if verbose:
            print("submission train data file [%s] generated (2015 dataset)." % (self.__out_trainDataFile.replace('.csv', '_2015.csv')))
        train_label[s.to_csv(self.__out_trainLabelsFile.replace('.csv', '_2015.csv'))
        if verbose:
            print("submission train labels file [%s] generated (2015 dataset)." % (self.__out_trainLabelsFile.replace('.csv', '_2015.csv')))
        '''
        
        '''
        if verbose:
            print("\nProducing SCALED output files now:\n---------------------------")
        test.to_csv(self.__out_testDataFile.replace('.csv', '_scaled.csv'))
        if verbose:
            print("submission test data file [%s] generated." % (self.__out_testDataFile.replace('.csv', '_scaled.csv')))
            
        train.to_csv(self.__out_trainDataFile.replace('.csv', '_2015_scaled.csv'))
        if verbose:
            print("submission train data file [%s] generated (2015 dataset)." % (self.__out_trainDataFile.replace('.csv', '_2015_scaled.csv')))
        '''
        
        '''
        train.loc[dates > 23].to_csv(self.__out_trainDataFile.replace('.csv', '_2015.csv'))
        if verbose:
            print("submission train data file [%s] generated (SMALL dataset)." % (self.__out_trainDataFile.replace('.csv', '_2015.csv')))
        train_labels[dates > 23].to_csv(self.__out_trainLabelsFile.replace('.csv', '_2015.csv'))
        if verbose:
            print("submission train labels file [%s] generated (SMALL dataset)." % (self.__out_trainLabelsFile.replace('.csv', '_2015.csv')))
        '''
        '''
        train.to_csv(self.__out_trainDataFile)
        if verbose:
            print("submission train data file [%s] generated (ENTIRE dataset)." % (self.__out_trainDataFile))
        train_labels.to_csv(self.__out_trainLabelsFile)
        if verbose:
            print("submission train labels file [%s] generated (ENTIRE dataset)." % (self.__out_trainLabelsFile))
        '''
        '''
        train.loc[dates > 11].to_csv(self.__out_trainDataFile.replace('.csv', '_MEDIUM.csv'))
        if verbose:
            print("submission train data file [%s] generated (MEDIUM dataset)." % (self.__out_trainDataFile.replace('.csv', '_MEDIUM.csv')))
        train_labels.loc[dates > 11].to_csv(self.__out_trainLabelsFile.replace('.csv', '_MEDIUM.csv'))
        if verbose:
            print("submission train labels file [%s] generated (MEDIUM dataset)." % (self.__out_trainLabelsFile.replace('.csv', '_MEDIUM.csv')))

        train.loc[dates > 21].to_csv(self.__out_trainDataFile.replace('.csv', '_SMALL.csv'))
        if verbose:
            print("submission train data file [%s] generated (SMALL dataset)." % (self.__out_trainDataFile.replace('.csv', '_SMALL.csv')))
        train_labels[dates > 21].to_csv(self.__out_trainLabelsFile.replace('.csv', '_SMALL.csv'))
        if verbose:
            print("submission train labels file [%s] generated (SMALL dataset)." % (self.__out_trainLabelsFile.replace('.csv', '_SMALL.csv')))
        '''
        

        return
