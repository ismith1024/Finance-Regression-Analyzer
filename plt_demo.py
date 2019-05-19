import pandas as pd
import sqlite3
import datetime
import numpy as np
import matplotlib
import sys
from matplotlib import pyplot as plt
#plt.figure(figsize=(20,10))

kern_200 = [0.000001,0.000001,0.000001,0.000001,0.000001,0.000002,0.000002,0.000002,0.000003,0.000003,0.000004,0.000005,0.000006,0.000007,0.000009,0.00001,0.000012,0.000015,0.000017,0.000021,0.000024,0.000029,0.000034,0.00004,0.000047,0.000054,0.000063,0.000074,0.000086,0.000099,0.000115,0.000133,0.000153,0.000176,0.000202,0.000231,0.000264,0.000301,0.000342,0.000388,0.00044,0.000498,0.000562,0.000632,0.000711,0.000797,0.000892,0.000996,0.00111,0.001235,0.001371,0.001519,0.001679,0.001852,0.002038,0.002239,0.002455,0.002686,0.002932,0.003194,0.003473,0.003769,0.00408,0.004409,0.004754,0.005116,0.005494,0.005888,0.006297,0.00672,0.007157,0.007607,0.008068,0.00854,0.00902,0.009508,0.010002,0.010499,0.010999,0.011498,0.011996,0.012489,0.012975,0.013453,0.013919,0.014372,0.014809,0.015228,0.015626,0.016002,0.016353,0.016677,0.016972,0.017237,0.01747,0.01767,0.017835,0.017964,0.018058,0.018114,0.018132,0.018114,0.018058,0.017964,0.017835,0.01767,0.01747,0.017237,0.016972,0.016677,0.016353,0.016002,0.015626,0.015228,0.014809,0.014372,0.013919,0.013453,0.012975,0.012489,0.011996,0.011498,0.010999,0.010499,0.010002,0.009508,0.00902,0.00854,0.008068,0.007607,0.007157,0.00672,0.006297,0.005888,0.005494,0.005116,0.004754,0.004409,0.00408,0.003769,0.003473,0.003194,0.002932,0.002686,0.002455,0.002239,0.002038,0.001852,0.001679,0.001519,0.001371,0.001235,0.00111,0.000996,0.000892,0.000797,0.000711,0.000632,0.000562,0.000498,0.00044,0.000388,0.000342,0.000301,0.000264,0.000231,0.000202,0.000176,0.000153,0.000133,0.000115,0.000099,0.000086,0.000074,0.000063,0.000054,0.000047,0.00004,0.000034,0.000029,0.000024,0.000021,0.000017,0.000015,0.000012,0.00001,0.000009,0.000007,0.000006,0.000005,0.000004,0.000003,0.000003,0.000002,0.000002,0.000002,0.000001,0.000001,0.000001,0.000001,0.000001]
kern_50 =  [0,0.000001,0.000002,0.000005,0.000012,0.000027,0.00006,0.000125,0.000251,0.000484,0.000898,0.001601,0.002743,0.004514,0.00714,0.010852,0.015849,0.022242,0.029993,0.038866,0.048394,0.057904,0.066574,0.073551,0.078084,0.079656,0.078084,0.073551,0.066574,0.057904,0.048394,0.038866,0.029993,0.022242,0.015849,0.010852,0.00714,0.004514,0.002743,0.001601,0.000898,0.000484,0.000251,0.000125,0.00006,0.000027,0.000012,0.000005,0.000002,0.000001,0]


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
    gain = last_close / row['close']
    ann_gain = gain ** (1/elapsed_years)
    return 100 * (ann_gain - 1.0) 

def main():
    if len(sys.argv) == 2:
        symbol = sys.argv[1]

        yahoo_db = '/home/ian/Data/yahoo.db'
        tmx_db = '/home/ian/Data/tmx.db'
        advfn_db = '/home/ian/Data/advfn.db'

        yahoo_database = sqlite3.connect(yahoo_db)
        tmx_database = sqlite3.connect(tmx_db)
        advfn_database = sqlite3.connect(advfn_db)
        #symbol = 'SLF'

        print('Generate data set for {0}'.format(symbol))
        print('   SQL queries...')

        tmx_sql = '''SELECT date, eps FROM tmx_earnings WHERE symbol = "{0}"'''.format(symbol)
        df_tmx = pd.read_sql_query(tmx_sql, tmx_database)
        df_tmx.columns = ['date', 'eps']
        df_tmx['date_parsed'] = df_tmx['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df_tmx.drop(columns = 'date', inplace = True)

        aav_sql = '''SELECT Date, Close FROM aav_prices WHERE symbol = "{0}" AND close != "null"'''.format(symbol)
        df_aav = pd.read_sql_query(aav_sql, yahoo_database)
        df_aav.columns = ['date', 'close']
        df_aav['date_parsed'] = df_aav['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df_aav.drop(columns = 'date', inplace = True)

        yahoo_prices_sql = '''SELECT Date, Close FROM tsx_prices WHERE symbol = "{0}" AND close != "null"'''.format(symbol)
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

        df.plot(x = 'date_parsed', y ='avg_50', figsize=(20,10), title='Daily Close')
        df[:'2018-06-01'].plot(x = 'date_parsed', y ='tot_gain', figsize=(20,10), title='Total Return')
        df.hist(['pe'], bins=40, figsize=(20,10))
        df.plot(x = 'date_parsed', y ='pe', figsize=(20,10), title = 'P-E Time Series')

        if divs:
            df.hist(['dy'], bins=40, figsize=(20,10))
            df.plot(x = 'date_parsed', y ='dy', figsize=(20,10), title = 'Dividend Time Series')


        plt.show()

    else:
        print('Please provide the symbol')


if __name__ == '__main__':
    main()