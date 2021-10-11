# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:37:12 2019

@author: herow
"""
import pandas as pd
import requests
import json

class Lev(object):
    '''get the DataFRame into lev DF, pp(qq or mm) DF, yy DF'''
    def __init__(self, df, yy_interval, pp_interval=1):
        self.lv = df
        self.yy = self.lv.pct_change(periods = yy_interval).dropna()
        self.yy = self.yy.add_suffix('_yy')
        self.pp = self.lv.pct_change(periods = pp_interval).dropna()
        if yy_interval == 12:
            self.pp = self.pp.add_suffix('_mm')
        elif yy_interval == 4:
            self.pp = self.pp.add_suffix('_qq')
        self.all = self.lv.join([self.pp, self.yy])

class BLS_API(object):
    '''This class try to gather the data from BLS API, 
    Here is the Series id reference
    https://www.bls.gov/help/hlpforma.htm#OEUS
    Local Area Unemployment Statistics
    Survey Overview The following is a sample format description of the Local Area Unemployment Statistics' series identifier:
    	                      1         2
    	             12345678901234567890
    	Series ID    LAUCN281070000000003
    	Positions    Value            Field Name
    	1-2          LA               Prefix
    	3            U                Seasonal Adjustment Code
    	4-18         CN2810700000000  Area Code
    	19-20        03               Measure Code
        
    06 'Labor force',
    05 'Employment',
    04 'Unemployment',
    03 'Unemployment rate'.
    '''
    def __init__(self, series_id, startyear, endyear, path):
        self.series_id = series_id #Dallas MSA employment id
        self.startyear = startyear
        self.endyear = endyear  
        self.path = path
        self.yy_interval = 12
        
    def get_data(self):
        headers = {'Content-type': 'application/json'}
        data = json.dumps({"seriesid": self.series_id,"startyear":self.startyear, "endyear":self.endyear})
        p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
        json_data = json.loads(p.text)
        df_all = pd.DataFrame()
        for series in json_data['Results']['series']:
            if series['data']:
                df = pd.DataFrame(series['data'])
                df['value'] = pd.to_numeric(df['value'])
                df['date'] = df['year'] + df['periodName']
                df['date'] = pd.to_datetime(df['date'],format = '%Y%B')
                df = df[['date','value']].set_index('date')
                df.columns = [series['seriesID']]
                df_all = pd.concat([df_all,df], axis = 1)
        return df_all

    def out_data(self):
        '''output data and plot the chart in excel'''
        data = self.get_data()
        data = Lev(data, self.yy_interval)
        writer = pd.ExcelWriter(self.path, engine='xlsxwriter',datetime_format='YYYY-MM-DD')
        workbook = writer.book
        format1 = workbook.add_format({'num_format': '0.0%'})

        tab = 'data'
        data.all.to_excel(writer,tab)
        worksheet = writer.sheets[tab]
        worksheet.set_column('A:A', 12)
        worksheet.set_column('C:D', 10, format1)
        # Create a new chart object. In this case an embedded chart.
        chart1 = workbook.add_chart({'type': 'line'})
        
        # Configure the first series.
        chart1.add_series({
            'name':       '='+tab+'!$D$1',
            'categories': '='+tab+'!$A$14:$A$'+str(data.all.size+1),
            'values':     '='+tab+'!$D$14:$D$'+str(data.all.size+1),
        })
        
        # Add a chart title and some axis labels.
        chart1.set_title ({'name': 'Dallas MSA Employment YoY Change'})
        chart1.set_x_axis({'name': 'month'})
        chart1.set_y_axis({'name': 'YoY Change'})
        
        # Turn off chart legend. It is on by default in Excel.
        chart1.set_legend({'position': 'none'})
        # Insert the chart into the worksheet (with an offset).
        worksheet.insert_chart('E2', chart1)

#        workbook.close()
        writer.save()
        writer.close()
    
if __name__ == '__main__':
    series_id = ['LAUMT481910000000005'] #Dallas MSA employment id
    startyear = 2013
    endyear = 2019  
    path = 'C:\\Users\\herow\\dallas_employment_yy.xlsx'
    bls = BLS_API(series_id, startyear, endyear, path)
    bls.out_data()


