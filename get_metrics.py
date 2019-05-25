import pandas as pd
import sqlite3
import datetime
import numpy as np
import matplotlib
import sys
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt
#plt.figure(figsize=(20,10))

kern_200 = [0.000001,0.000001,0.000001,0.000001,0.000001,0.000002,0.000002,0.000002,0.000003,0.000003,0.000004,0.000005,0.000006,0.000007,0.000009,0.00001,0.000012,0.000015,0.000017,0.000021,0.000024,0.000029,0.000034,0.00004,0.000047,0.000054,0.000063,0.000074,0.000086,0.000099,0.000115,0.000133,0.000153,0.000176,0.000202,0.000231,0.000264,0.000301,0.000342,0.000388,0.00044,0.000498,0.000562,0.000632,0.000711,0.000797,0.000892,0.000996,0.00111,0.001235,0.001371,0.001519,0.001679,0.001852,0.002038,0.002239,0.002455,0.002686,0.002932,0.003194,0.003473,0.003769,0.00408,0.004409,0.004754,0.005116,0.005494,0.005888,0.006297,0.00672,0.007157,0.007607,0.008068,0.00854,0.00902,0.009508,0.010002,0.010499,0.010999,0.011498,0.011996,0.012489,0.012975,0.013453,0.013919,0.014372,0.014809,0.015228,0.015626,0.016002,0.016353,0.016677,0.016972,0.017237,0.01747,0.01767,0.017835,0.017964,0.018058,0.018114,0.018132,0.018114,0.018058,0.017964,0.017835,0.01767,0.01747,0.017237,0.016972,0.016677,0.016353,0.016002,0.015626,0.015228,0.014809,0.014372,0.013919,0.013453,0.012975,0.012489,0.011996,0.011498,0.010999,0.010499,0.010002,0.009508,0.00902,0.00854,0.008068,0.007607,0.007157,0.00672,0.006297,0.005888,0.005494,0.005116,0.004754,0.004409,0.00408,0.003769,0.003473,0.003194,0.002932,0.002686,0.002455,0.002239,0.002038,0.001852,0.001679,0.001519,0.001371,0.001235,0.00111,0.000996,0.000892,0.000797,0.000711,0.000632,0.000562,0.000498,0.00044,0.000388,0.000342,0.000301,0.000264,0.000231,0.000202,0.000176,0.000153,0.000133,0.000115,0.000099,0.000086,0.000074,0.000063,0.000054,0.000047,0.00004,0.000034,0.000029,0.000024,0.000021,0.000017,0.000015,0.000012,0.00001,0.000009,0.000007,0.000006,0.000005,0.000004,0.000003,0.000003,0.000002,0.000002,0.000002,0.000001,0.000001,0.000001,0.000001,0.000001]
kern_50 =  [0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0]

yahoo_db = '/home/ian/Data/yahoo.db'
tmx_db = '/home/ian/Data/tmx.db'
advfn_db = '/home/ian/Data/advfn.db'

yahoo_database = sqlite3.connect(yahoo_db)
tmx_database = sqlite3.connect(tmx_db)
advfn_database = sqlite3.connect(advfn_db)
advfn_curs = advfn_database.cursor()

def process_row(row, df):
    if (row['close'] != row['close']) | (row['close'] == 'null'):
        df.at[row['date_parsed'], 'pe'] = np.NaN
        df.at[row['date_parsed'], 'dy'] = np.NaN
        return
    if row['eps'] == row['eps']:
        if row['eps'] == 0:
            df.at[row['date_parsed'], 'pe'] = np.NaN
        else:
            df.at[row['date_parsed'], 'pe'] = float(row['close']) / (4*row['eps'])
    
    if row['eps'] == row['eps']:
        if row['close'] == 0:
            df.at[row['date_parsed'], 'dy'] = np.NaN
        else:
            df.at[row['date_parsed'], 'dy'] = 4* row['div'] / (float(row['close']) * row['split_adj'])

def custom_kernel(func, kern):
    '''
    Convolution of a function by a kernel.
    Kernel must be odd in length
    Function must be longer than kernel    
    '''
    if len(func) < len(kern):
        return func
    else:
        #midpoint of the kernel
        mid_kern = int((len(kern) + 1)/2)
        
    conv_func = np.zeros(len(func))
    
    for index, value in enumerate(func):
        
        ##TODO: this case is backwards I think
        if index < mid_kern:
            #go from kern[mid_kern] to end for zero
            dist_from_start = index -1
            kern_start = mid_kern - dist_from_start         
           
            sum = 0.0
            area = 0.0
            for i in range(kern_start, len(kern)):                
                sum += kern[i] * func[index + i - mid_kern]
                #need to correct for the area under the partial kernel being < 1
                area += kern[i]
            conv_func[index] = sum / area
            
        elif index > len(func) - mid_kern:         
            #go from zero to mid_kern + (distance to end of func)
            dist_to_end = len(func) - index + 1
            sum = 0.0
            area = 0.0
            for i in range(0, (mid_kern + dist_to_end -1)):
                sum += kern[i] * func[index + i - mid_kern]
                area += kern[i]
            conv_func[index] = sum / area   
            
        else:
            #sum of kernel * function over window of kernel length centered on func[index]
            sum = 0.0
            for kern_ind, kern_val in enumerate(kern):
                sum += kern_val * func[index + kern_ind - mid_kern]
            conv_func[index] = sum
        
    return conv_func

def return_to_date(row, today, last_close):
    elapsed_years = (today - row['date_parsed']).days / 365.25
    if elapsed_years == 0:
        return 1.0
    gain = last_close / row['close']
    ann_gain = gain ** (1/elapsed_years)
    return 100 * (ann_gain - 1.0)

def print_metrics(df, divs):
    '''
    Plots the 50-day smoothed price series, total return, p-e & yield histograms, and p-e & yeild time series
    Only plots dividends if divs == True
    '''
    #df.plot(x = 'date_parsed', y ='avg_50', figsize=(20,10), title='Daily Close')
    #df[:'2018-06-01'].plot(x = 'date_parsed', y ='tot_gain', figsize=(20,10), title='Total Return')
    #df.hist(['pe'], bins=40, figsize=(20,10))
    #df.plot(x = 'date_parsed', y ='pe', figsize=(20,10), title = 'P-E Time Series')

    #if divs:
    #    df.hist(['dy'], bins=40, figsize=(20,10))
    #    df.plot(x = 'date_parsed', y ='dy', figsize=(20,10), title = 'Dividend Time Series')

    #plt.show()

def prune_data(df, divs, num_sig):
    '''
    Removes outliers from the dataframe
    Removes num_sig-sigma
    '''

    if divs:
        pe_mean = np.mean(df['pe'])
        pe_std = np.std(df['pe'])
        div_mean = np.mean(df['dy'])
        div_std = np.std(df['dy'])
        gain_mean = np.mean(df['tot_gain'])
        gain_std = np.std(df['tot_gain'])
        pe_upper = pe_mean + num_sig * pe_std
        pe_lower = pe_mean - num_sig * pe_std
        div_upper = div_mean + num_sig * div_std
        div_lower = div_mean - num_sig * div_std
        gain_upper = gain_mean + num_sig * gain_std
        gain_lower = gain_mean - num_sig * gain_std

        df_pruned = df[(df['pe'] < pe_upper) & (df['pe'] > pe_lower) & (df['dy'] < div_upper) & (df['dy'] > div_lower) & (df['pe'] > 0) & (df['tot_gain'] < gain_upper) & (df['tot_gain'] > gain_lower)].copy()
        #df_pruned = df[(df['pe'] < pe_upper)].copy()
 
        #print('P-E mean: {0}  P-E std dev: {1}  P-E Upper: {2}  P-E Lower: {3}'.format(str(pe_mean), str(pe_std), str(pe_upper), str(pe_lower)))

    else:
        pe_mean = np.mean(df['pe'])
        pe_std = np.std(df['pe'])

        pe_upper = pe_mean + num_sig * pe_std
        pe_lower = pe_mean - num_sig * pe_std
        
        gain_mean = np.mean(df['tot_gain'])
        gain_std = np.std(df['tot_gain'])
        
        gain_upper = gain_mean + num_sig * gain_std
        gain_lower = gain_mean - num_sig * gain_std

        df_pruned = df[(df['pe'] < pe_upper) & (df['pe'] > pe_lower) & (df['pe'] > 0) & (df['tot_gain'] < gain_upper) & (df['tot_gain'] > gain_lower)].copy()

    #df_pruned.hist(['pe'], bins=50, figsize=(20,10))

    #if divs:
    #    df_pruned.hist(['dy'], bins=50, figsize=(20,10))
        
    #plt.xticks(())
    #plt.yticks(())
    #plt.show()
    return df_pruned

def show_regression(df, divs, symbol):
    '''
    Trains a regression learner for pe-total gain and yield-total gain
    K-folds the data and prints r_squared for each fold
    Plots the most recent fold
    Gain is used up to 2017-05-06
    '''

    #was 2018-05-06

    df_pruned = prune_data(df[:'2017-05-06'], divs, 3.0)

    if df_pruned.shape[0] < 3:
        print('Not enough data')
        return

    X = pd.DataFrame(df_pruned['pe'])
    y = pd.DataFrame(df_pruned['tot_gain'])

    X.fillna(value = 0, inplace = True)
    y.fillna(value = 0, inplace = True)

    model = LinearRegression()
    scores = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kfold.split(X, y)):
        model.fit(X.iloc[train,:], y.iloc[train,:])
        score = model.score(X.iloc[test,:], y.iloc[test,:])
        scores.append(score)

    print('Rsquared for 3-fold p-e data:' + str(scores))
    ### Plot the regression
    y_pred = model.predict(X)
    #plt.scatter(X, y,  color='black')
    #plt.plot(X, y_pred, color='blue', linewidth=3)
    #plt.xticks(())
    #plt.yticks(())
    #plt.show()

    sql_text = 'INSERT OR IGNORE INTO regression_metrics(symbol, method, fold_1, fold_2, fold_3) VALUES(?, ?, ?, ?, ?)'
    job = (symbol, 'p-e', scores[0], scores[1], scores[2])
    advfn_curs.execute(sql_text, job)
    advfn_database.commit()

    if divs:
        X = pd.DataFrame(df_pruned['dy'])
        y = pd.DataFrame(df_pruned['tot_gain'])

        X.fillna(value = 0, inplace = True)
        y.fillna(value = 0, inplace = True)

        model2 = LinearRegression()
        scores2 = []
        kfold2 = KFold(n_splits=3, shuffle=True, random_state=42)
        for i, (train, test) in enumerate(kfold2.split(X, y)):
            model2.fit(X.iloc[train,:], y.iloc[train,:])
            score2 = model2.score(X.iloc[test,:], y.iloc[test,:])
            scores2.append(score2)

        print('Rsquared for 3-fold div data:' + str(scores2))

        sql_text = 'INSERT OR IGNORE INTO regression_metrics(symbol, method, fold_1, fold_2, fold_3) VALUES(?, ?, ?, ?, ?)'
        job = (symbol, 'divs', scores2[0], scores2[1], scores2[2])
        advfn_curs.execute(sql_text, job)
        advfn_database.commit()

        y_pred2 = model2.predict(X)
        #plt.scatter(X, y,  color='black')
        #plt.plot(X, y_pred2, color='blue', linewidth=3)
        #plt.xticks(())
        #plt.yticks(())
        #plt.show()

    #df_pruned.hist(['tot_gain'], bins=50, figsize=(20,10))
    #plt.xticks(())
    #plt.yticks(())
    #plt.show()

def show_regression2(df, divs, symbol):
    '''
    Trains a regression learner for pe-total gain and yield-total gain
    K-folds the data and prints r_squared for each fold
    Plots the most recent fold
    Gain is used up to 2017-05-06
    '''

    #was 2018-05-06

    df_pruned = prune_data(df[:'2017-05-06'], divs, 3.0)

    if df_pruned.shape[0] < 3:
        print('Not enough data')
        return

    if not divs:

        X = pd.DataFrame(df_pruned['pe'])
        y = pd.DataFrame(df_pruned['tot_gain'])

        X.fillna(value = 0, inplace = True)
        y.fillna(value = 0, inplace = True)

        model = LinearRegression()
        scores = []
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        for i, (train, test) in enumerate(kfold.split(X, y)):
            model.fit(X.iloc[train,:], y.iloc[train,:])
            score = model.score(X.iloc[test,:], y.iloc[test,:])
            scores.append(score)

        print('Rsquared for 3-fold p-e data:' + str(scores))
        ### Plot the regression
        y_pred = model.predict(X)
        #plt.scatter(X, y,  color='black')
        #plt.plot(X, y_pred, color='blue', linewidth=3)
        #plt.xticks(())
        #plt.yticks(())
        #plt.show()

    else:
        X = pd.DataFrame(df_pruned[['pe', 'dy']])
        y = pd.DataFrame(df_pruned['tot_gain'])

        X.fillna(value = 0, inplace = True)
        y.fillna(value = 0, inplace = True)

        model2 = LinearRegression()
        scores2 = []
        kfold2 = KFold(n_splits=3, shuffle=True, random_state=42)
        for i, (train, test) in enumerate(kfold2.split(X, y)):
            model2.fit(X.iloc[train,:], y.iloc[train,:])
            score2 = model2.score(X.iloc[test,:], y.iloc[test,:])
            scores2.append(score2)

        print('Rsquared for 3-fold combined data:' + str(scores2))

        y_pred2 = model2.predict(X)
        
        #plt.scatter(X['dy'], y,  color='blue')
        #plt.plot(X['dy'], y_pred2, color='red', linewidth=3)
        #plt.xticks(())
        #plt.yticks(())
        #plt.show()

        #plt.scatter(X['pe'], y,  color='blue')
        #plt.plot(X['pe'], y_pred2, color='red', linewidth=3)
        #plt.xticks(())
        #plt.yticks(())
        #plt.show()

        sql_text = 'INSERT OR IGNORE INTO regression_metrics(symbol, method, fold_1, fold_2, fold_3) VALUES(?, ?, ?, ?, ?)'
        job = (symbol, 'multiple', scores2[0], scores2[1], scores2[2])
        advfn_curs.execute(sql_text, job)
        advfn_database.commit()

    #df_pruned.hist(['tot_gain'], bins=50, figsize=(20,10))
    #plt.xticks(())
    #plt.yticks(())
    #plt.show()


'''
advfn.db - uses google notation
tmx.db - uses google notation
yahoo.db 
    aav_prices uses yahoo notation
    divs uses yahoo notation
    splits uses yahoo notation
    tsx_prices uses google notation
    yahoo_indicators uses yahoo notation

'''

def main():
    if len(sys.argv) == 2:
        symbol = sys.argv[1]

        print('Generate data set for {0}'.format(symbol))
        print('   SQL queries...')

        #change this from yahoo to google notation
        tmx_sql = '''SELECT date, eps FROM tmx_earnings WHERE symbol = "{0}"'''.format(symbol.replace('-','.'))
        df_tmx = pd.read_sql_query(tmx_sql, tmx_database)
        df_tmx.columns = ['date', 'eps']
        df_tmx['date_parsed'] = df_tmx['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df_tmx.drop(columns = 'date', inplace = True)

        aav_sql = '''SELECT Date, Close FROM aav_prices WHERE symbol = "{0}" AND close != "null"'''.format(symbol)
        df_aav = pd.read_sql_query(aav_sql, yahoo_database)
        df_aav.columns = ['date', 'close']
        df_aav['date_parsed'] = df_aav['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df_aav.drop(columns = 'date', inplace = True)

        #need to change yahoo notation to google notation
        yahoo_prices_sql = '''SELECT Date, Close FROM tsx_prices WHERE symbol = "{0}" AND close != "null"'''.format(symbol.replace('-','.'))
        df_y_price = pd.read_sql_query(yahoo_prices_sql, yahoo_database)
        df_y_price.columns = ['date', 'close']
        df_y_price['date_parsed'] = df_y_price['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df_y_price.drop(columns = 'date', inplace = True)

        divs_sql = '''SELECT Date, Dividends FROM divs WHERE symbol = "{0}"'''.format(symbol)
        df_divs = pd.read_sql_query(divs_sql, yahoo_database) 
        df_divs.columns = ['date', 'div']
        df_divs['date_parsed'] = df_divs['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df_divs.drop(columns = 'date', inplace = True)

        df_price = pd.concat([df_y_price, df_aav])
        print('Before: ' + str(df_price.shape[0]))
        df_price.drop_duplicates(subset='date_parsed', inplace = True)
        print('After: ' + str(df_price.shape[0]))
        df_price

        split_sql = '''SELECT date, total_adjustment FROM splits WHERE symbol = "{0}"'''.format(symbol)
        df_split = pd.read_sql_query(split_sql, yahoo_database) 
        df_split.columns = ['date', 'split_adj']
        df_split['date_parsed'] = df_split['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df_split.drop(columns = 'date', inplace = True)

        print('   complete!')

        print('   Merge dataframes...')
        df = df_price.join(df_tmx.set_index('date_parsed'), on = 'date_parsed', how = 'outer', sort = True)

        df = df.join(df_divs.set_index('date_parsed'), on = 'date_parsed', how = 'outer', sort = True)

        df.fillna(method='ffill', inplace = True)

        df = df.join(df_split.set_index('date_parsed'), on = 'date_parsed', how = 'outer', sort = True)
        df.set_index(df['date_parsed'], inplace= True)

        #split adjustment for current date is 1.0 -- backfill missing values

        df.iloc[-1, df.columns.get_loc('split_adj')] = 1.0
        df['split_adj'].fillna(method='bfill', inplace = True)
        df.tail(5)

        #this is for current quarter only - go back and fill the TTM on df_earnings and df_divs
        df['pe'] = 0.0
        df['dy'] = 0.0

        print('   complete!')
        print('   Calculate yield and eps...')
            
            
        df['close'].fillna(method = 'bfill', inplace = True)
        df.apply((lambda x: process_row(x, df)), axis = 1)

        print('   complete!')
        print('   Calculate returns...')

        df['avg_50'] = custom_kernel(df['close'], kern_50)
        df['avg_200'] = custom_kernel(df['close'], kern_200)

        today = datetime.datetime.today()
        last_close = df.tail(1)['avg_200'][0]

        #print('Elapsed: ' + str((today - df.head(1)['date_parsed'][0]).days / 365.25 ) + ' years')

        df['cap_gain'] = df.apply(lambda x: return_to_date(x, today, last_close), axis = 1) 
        if (df.tail(1)['dy'][0] > 0):
            divs = True
            print('Found a dividend yield')
            df['tot_gain'] = df['cap_gain'] + (df['dy'] * 100)
        else:
            divs = False
            print('No dividend yield found')
            df['tot_gain'] = df['cap_gain']

        print('  complete!')

        #print_metrics(df, divs)
        show_regression(df, divs, symbol)
        show_regression2(df, divs, symbol)
        #prune_data(df, divs, 3.0)
        

    else:
        sql_symbols = 'SELECT company_ticker FROM tsx_companies'
        advfn_curs.execute(sql_symbols)
        res = advfn_curs.fetchall()
        symbols = []
        for sym in res:
            symbols.append(sym[0])
            symbol = sym[0]
            print('Generate data set for {0}'.format(symbol))
            print('   SQL queries...')

            #change this from yahoo to google notation
            tmx_sql = '''SELECT date, eps FROM tmx_earnings WHERE symbol = "{0}"'''.format(symbol.replace('-','.'))
            df_tmx = pd.read_sql_query(tmx_sql, tmx_database)
            df_tmx.columns = ['date', 'eps']
            df_tmx['date_parsed'] = df_tmx['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            df_tmx.drop(columns = 'date', inplace = True)

            aav_sql = '''SELECT Date, Close FROM aav_prices WHERE symbol = "{0}" AND close != "null"'''.format(symbol)
            df_aav = pd.read_sql_query(aav_sql, yahoo_database)
            df_aav.columns = ['date', 'close']
            df_aav['date_parsed'] = df_aav['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            df_aav.drop(columns = 'date', inplace = True)

            #need to change yahoo notation to google notation
            yahoo_prices_sql = '''SELECT Date, Close FROM tsx_prices WHERE symbol = "{0}" AND close != "null"'''.format(symbol.replace('-','.'))
            df_y_price = pd.read_sql_query(yahoo_prices_sql, yahoo_database)
            df_y_price.columns = ['date', 'close']
            df_y_price['date_parsed'] = df_y_price['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            df_y_price.drop(columns = 'date', inplace = True)

            divs_sql = '''SELECT Date, Dividends FROM divs WHERE symbol = "{0}"'''.format(symbol)
            df_divs = pd.read_sql_query(divs_sql, yahoo_database) 
            df_divs.columns = ['date', 'div']
            df_divs['date_parsed'] = df_divs['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            df_divs.drop(columns = 'date', inplace = True)

            df_price = pd.concat([df_y_price, df_aav])
            print('Before: ' + str(df_price.shape[0]))
            df_price.drop_duplicates(subset='date_parsed', inplace = True)
            print('After: ' + str(df_price.shape[0]))
            df_price

            split_sql = '''SELECT date, total_adjustment FROM splits WHERE symbol = "{0}"'''.format(symbol)
            df_split = pd.read_sql_query(split_sql, yahoo_database) 
            df_split.columns = ['date', 'split_adj']
            df_split['date_parsed'] = df_split['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            df_split.drop(columns = 'date', inplace = True)

            print('   complete!')

            print('   Merge dataframes...')
            df = df_price.join(df_tmx.set_index('date_parsed'), on = 'date_parsed', how = 'outer', sort = True)

            df = df.join(df_divs.set_index('date_parsed'), on = 'date_parsed', how = 'outer', sort = True)

            df.fillna(method='ffill', inplace = True)

            df = df.join(df_split.set_index('date_parsed'), on = 'date_parsed', how = 'outer', sort = True)
            df.set_index(df['date_parsed'], inplace= True)

            #split adjustment for current date is 1.0 -- backfill missing values

            df.iloc[-1, df.columns.get_loc('split_adj')] = 1.0
            df['split_adj'].fillna(method='bfill', inplace = True)
            df.tail(5)

            #this is for current quarter only - go back and fill the TTM on df_earnings and df_divs
            df['pe'] = 0.0
            df['dy'] = 0.0

            print('   complete!')
            print('   Calculate yield and eps...')
                
                
            df['close'].fillna(method = 'bfill', inplace = True)
            df.apply((lambda x: process_row(x, df)), axis = 1)

            print('   complete!')
            print('   Calculate returns...')

            df['avg_50'] = custom_kernel(df['close'], kern_50)
            df['avg_200'] = custom_kernel(df['close'], kern_200)

            today = datetime.datetime.today()
            last_close = df.tail(1)['avg_200'][0]

            #print('Elapsed: ' + str((today - df.head(1)['date_parsed'][0]).days / 365.25 ) + ' years')

            df['cap_gain'] = df.apply(lambda x: return_to_date(x, today, last_close), axis = 1) 
            if (df.tail(1)['dy'][0] > 0):
                divs = True
                print('Found a dividend yield')
                df['tot_gain'] = df['cap_gain'] + (df['dy'] * 100)
            else:
                divs = False
                print('No dividend yield found')
                df['tot_gain'] = df['cap_gain']

            print('  complete!')

            #print_metrics(df, divs)
            show_regression(df, divs, symbol)
            show_regression2(df, divs, symbol)
            #prune_data(df, divs, 3.0)



if __name__ == '__main__':
    main()