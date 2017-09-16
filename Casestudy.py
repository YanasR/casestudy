import pandas as pd
import os
import matplotlib.pyplot as plt
from _datetime import datetime
import numpy as np

from prediction_methods import predict_sales_on_trans


def main():
   
    Reading_file()
     

def Reading_file():
    
 
    current_file = os.path.abspath(os.path.dirname(__file__)) #older/folder2/scripts_folder

#csv_filename
    csv_filename = os.path.join(current_file, '../Data/D11-02/D_combined')
    Data = pd.read_csv(csv_filename,sep=';')
    
    # Converting to date format Pandas
        
    Data['trans_date_time'] = pd.to_datetime(Data['trans_date_time'])
    
    plot_salesperday(Data)
    plot_transperday(Data)
    plot_transperdayhist(Data)
    predict_sales_on_trans(Data)
    
    
    
  #  print (Data['trans_date_time'].dt.weekday_name)

def plot_salesperday(Data):
    
   
    y_axis =Data[['trans_date_time','price']].groupby('trans_date_time')['price'].sum().tolist()
    
    x_axis =Data[['trans_date_time','price']].groupby('trans_date_time')['price'].sum().index.tolist()
 #   x =list(Data[['trans_date_time','price']].groupby('trans_date_time')['price'].sum().index.values)
   
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, 'o-')
    fig.autofmt_xdate()
    plt.show()
    plt.close()
    
def plot_transperday(Data):
    
   
   
    y_axis =Data['trans_date_time'].value_counts().sort_index().tolist()
    
    x_axis =Data['trans_date_time'].value_counts().sort_index().index.tolist()
    
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis, 'o-')
    fig.autofmt_xdate()
    plt.show()
    
def plot_transperdayhist(Data):
    
   

    Data =Data[['trans_date_time','price']].groupby('trans_date_time',as_index=False)['price'].sum()
    
    Data['trans_date_time'] = Data['trans_date_time'].dt.weekday_name
#    Data = Data.sort_values(by=['trans_date_time'])
    Data =Data[['trans_date_time','price']].groupby('trans_date_time',as_index=False)['price'].sum()
    
    Data = Data.sort_values(by=['price'])
    
   
    
   # Data.plot('trans_date_time','price',kind='bar',color='b')
    fig, ax = plt.subplots()
    
    bar_width = 0.4
    opacity = 0.6
    
    ax.bar(Data.index,Data['price'],width=0.6,color='b')
    ax.set_xticks(Data.index+0.20)
    ax.set_xticklabels(Data.trans_date_time)
 
    plt.show()
    
  

if __name__ == '__main__':
    main()
