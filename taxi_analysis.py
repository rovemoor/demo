# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:51:56 2019

@author: herow
"""

import pyodbc
import pandas as pd
import time
from random import randint
import numpy as np
import datetime as dt
from collections import Counter
import requests
from contextlib import closing

import gc
gc.collect()


date_rng = pd.date_range('2016-7-1', '2018-6-1', freq='MS')
date_rng_lst = date_rng.to_frame().iloc[:,0].apply(lambda x: x.strftime("%Y-%m")).tolist()

taxi_lst = ['yellow', 'green', 'fhv']
##### simply download of all the data line by line iteratively ########
try:
    for yrmm in date_rng_lst[:1]:
        for taxi in taxi_lst[:1]:
            url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'+taxi+'_tripdata_'+yrmm+'.csv'
            filename = 'D:\\case_DK\\data\\'+taxi+'_tripdata_'+yrmm+'_test.csv'

#            with urllib.request.urlopen(url) as testfile, open(filename, 'w') as f:
#                f.write(testfile.read().decode())
            with closing(requests.get(url, stream=True)) as response, open(filename, 'w') as f:
                lines = (line.decode('utf-8') for line in response.iter_lines())
                for l in lines: 
                        f.write(l+'\n')
            time.sleep(randint(1, 5))
except:
    print("loading err")

############ check the pattern of the data
for yymm in date_rng:
    yymm_str = yymm.strftime("%Y-%m")
    taxi = 'fhv'
    url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'+taxi+'_tripdata_'+yymm_str+'.csv'
    filename = 'D:\\case_DK\\data\\'+taxi+'_tripdata_'+yrmm+'_test.csv'

#            with urllib.request.urlopen(url) as testfile, open(filename, 'w') as f:
#                f.write(testfile.read().decode())
    with closing(requests.get(url, stream=True)) as response, open(filename, 'w') as f:
        lines = (line.decode('utf-8') for line in response.iter_lines())
        for i, l in enumerate(lines):
            if i <= 1 and l !='' :
                data = l.split(',')
                print(yymm_str, len(data), l)
            else:
                break
    time.sleep(randint(1, 5))

""" printing the first line of the FHV data shows 2016/7 - 2016/12 has only 3 cols,
2017/1 - 2017/6 has 5 cols
2017/7 - 2018/6 has 6 cols
"""
###### download the data and process all the stats line by line while reading and writing the data
yellow_hr_cnt = Counter()
green_hr_cnt = Counter()
fhv_hr_cnt = Counter()


fare = pd.DataFrame(0, index=date_rng, columns = taxi_lst)
user = pd.DataFrame(0, index=date_rng, columns = taxi_lst)
trip = pd.DataFrame(0, index=date_rng, columns = taxi_lst)

try:
    for yymm in date_rng:
#    for yymm in date_rng[-1:]:
        yymm_str = yymm.strftime("%Y-%m")
#        for taxi in taxi_lst:
        for taxi in taxi_lst[2:3]:
            temp_fare = 0
            temp_user = 0
            temp_trip = 0

            url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'+taxi+'_tripdata_'+yymm_str+'.csv'
            filename = 'D:\\case_DK\\data\\'+taxi+'_tripdata_'+yymm_str+'.csv'

            with closing(requests.get(url, stream=True)) as response, open(filename, 'w') as f:
                lines = (line.decode('utf-8') for line in response.iter_lines())

                if taxi =='green':
                    for i, l in enumerate(lines): 
                        f.write(l+'\n')
#                        print(l)
                        if i>= 2 and l !='' :
                            data = l.split(',')
                            # fare sum
                            temp = float(data[9])
                            if temp > 0: # if the fare > 0
                                temp_fare += temp
                                temp_user += int(data[7])
                                temp_trip += 1
                            # hourly freq
                            green_hr_cnt[dt.datetime.strptime(data[1], '%Y-%m-%d %H:%M:%S').replace(minute=0, second=0)] += 1
                    fare.loc[yymm, taxi] = temp_fare
                    user.loc[yymm, taxi] = temp_user
                    trip.loc[yymm, taxi] = temp_trip

                elif taxi =='yellow':
                    for i, l in enumerate(lines): 
                        f.write(l+'\n')
#                        print(l)
                        if i>= 2 and l !='' :
                            data = l.split(',')
                            # fare sum
                            temp = float(data[10])
                            if temp > 0: # if the fare > 0
                                temp_fare += temp
                                temp_user += int(data[3])
                                temp_trip += 1
                            # hourly freq
                            yellow_hr_cnt[dt.datetime.strptime(data[1], '%Y-%m-%d %H:%M:%S').replace(minute=0, second=0)] += 1
                    fare.loc[yymm, taxi] = temp_fare
                    user.loc[yymm, taxi] = temp_user     
                    trip.loc[yymm, taxi] = temp_trip                      

                elif taxi =='fhv':
                    # fhv only has 3 cols
                    if yymm >= pd.Timestamp('20160601') and yymm < pd.Timestamp('20170101'):
                        for i, l in enumerate(lines):
                            if i==0:
                                l='"Dispatching_base_num","Pickup_DateTime","DropOff_datetime","PUlocationID","DOlocationID","SR_Flag"'
                            elif l != '':
                                data = l.split(',')
                                data.insert(3,'')
                                data = data + ['' , '']
                                l = '"' + '","'.join(data) + '"'
                                f.write(l+'\n')
                                
                                temp_user += 1
                                temp_trip += 1
                                # hourly freq
                                fhv_hr_cnt[dt.datetime.strptime(data[1], '%Y-%m-%d %H:%M:%S').replace(minute=0, second=0)] += 1
                    
                    # fhv only has 3 cols
                    elif yymm >= pd.Timestamp('20170101') and yymm < pd.Timestamp('20170701'):
                        for i, l in enumerate(lines):
                            if i==0:
                                l='"Dispatching_base_num","Pickup_DateTime","DropOff_datetime","PUlocationID","DOlocationID","SR_Flag"'
                            elif l != '':
                                data = l.split(',')
                                data = data + ['']
                                l = '"' + '","'.join(data) + '"'
                                f.write(l+'\n')
                                
                                temp_user += 1
                                temp_trip += 1
                                # hourly freq
                                fhv_hr_cnt[dt.datetime.strptime(data[1], '%Y-%m-%d %H:%M:%S').replace(minute=0, second=0)] += 1
                    # fhv has 6cols
                    else:
                        for i, l in enumerate(lines):
                            f.write(l+'\n')
                            if i>= 1 and l !='' :
                                data = l.split('","')
                                # shared ride user = user + 2 if flag==1, else user = user + 1 
                                temp_user += 2 if data[5]=='1"' else 1
                                temp_trip += 1
                                # hourly freq
                                fhv_hr_cnt[dt.datetime.strptime(data[1], '%Y-%m-%d %H:%M:%S').replace(minute=0, second=0)] += 1
                    user.loc[yymm, taxi] = temp_user     
                    trip.loc[yymm, taxi] = temp_trip 

                    for i, l in enumerate(lines):
                        f.write(l+'\n')
#                        print(l)
                        if i>= 1 and l !='' :
                            data = l.split('","')
                            # shared ride user = user + 2 if flag==1, else user = user + 1 
                            temp_user += 2 if data[5]=='1"' else 1
                            temp_trip += 1
                            # hourly freq
                            fhv_hr_cnt[dt.datetime.strptime(data[1], '%Y-%m-%d %H:%M:%S').replace(minute=0, second=0)] += 1
                    user.loc[yymm, taxi] = temp_user     
                    trip.loc[yymm, taxi] = temp_trip   
                print('finish ', yymm_str, taxi)
            time.sleep(randint(1, 10))
except:
    print("loading err, stop at", yymm, taxi)
######### market share #######
discount = 0.7
fare_user = fare.copy()
fare_user['fhv'] = discount * (fare_user['yellow'] / user['yellow'] + fare_user['green'] / user['green']) / 2 * user['fhv']

fare_trip = fare.copy()
fare_trip['fhv'] = discount * (fare_trip['yellow'] / trip['yellow'] + fare_trip['green'] / trip['green']) / 2 * trip['fhv']


user.div(user.sum(1), axis=0).plot(figsize= (12,6),kind='bar', stacked=True, title = 'user market share')

trip.div(trip.sum(1), axis=0).plot(figsize= (12,6),kind='bar', stacked=True, title = 'trip market share')

fare_trip.div(fare_trip.sum(1), axis=0).plot(figsize= (12,6),kind='bar', stacked=True, title = 'fare market share')

##### hourly freq analysis
green_hr_cnt = pd.DataFrame.from_dict(green_hr_cnt, orient='index').sort_index()
green_hr_cnt.columns = ['green']
yellow_hr_cnt = pd.DataFrame.from_dict(yellow_hr_cnt, orient='index').sort_index()
yellow_hr_cnt.columns = ['yellow']
fhv_hr_cnt = pd.DataFrame.from_dict(fhv_hr_cnt, orient='index').sort_index()
fhv_hr_cnt.columns = ['fhv']
hr_cnt = green_hr_cnt.join([yellow_hr_cnt, fhv_hr_cnt])

hr_cnt = hr_cnt[(hr_cnt.index >='20160601') & (hr_cnt.index <'20180701')]
hr_cnt_all = hr_cnt.copy()
hr_cnt_all.plot(figsize= (12,6), title = 'all the hourly freq of the day')

hr_cnt.reset_index(inplace = True)
hr_cnt['hour']= hr_cnt['index'].apply(lambda x: x.hour)

mean = hr_cnt.groupby('hour').mean()
mean.plot(figsize= (12,6), title = 'hourly freq of the day')
stats_g = hr_cnt.groupby('hour')[['green']].apply(lambda x:x.describe()).unstack()
stats_g.iloc[:, 1:].plot(figsize= (12,6))

stats_y = hr_cnt.groupby('hour')[['yellow']].apply(lambda x:x.describe()).unstack()
stats_y.iloc[:, 1:].plot(figsize= (12,6))

stats_fhv = hr_cnt.groupby('hour')[['fhv']].apply(lambda x:x.describe()).unstack()
stats_fhv.iloc[:, 1:].plot(figsize= (12,6))





