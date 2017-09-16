import pandas as pd
import os
import matplotlib.pyplot as plt
from _datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import feature_selection
import matplotlib.pyplot as py
from mpl_toolkits.mplot3d import Axes3D


def main():
   
    print('testing')
     

    
def predict_sales_on_trans(Data):
    
   
     
    x_axis1 =Data['trans_date_time'].value_counts().sort_index().tolist()
    x_axis2 =Data[['trans_date_time','amount']].groupby('trans_date_time')['amount'].sum().tolist()
    
    y_axis =Data[['trans_date_time','price']].groupby('trans_date_time')['price'].sum().tolist()
    
    x_axis1 = np.array(x_axis1)
    x_axis2 = np.array(x_axis2)
    y_axis = np.array(y_axis)
      
    x_axis = np.column_stack((x_axis1,x_axis2))
    
    data = np.column_stack((x_axis,y_axis))
    
    X_train, X_test, y_train, y_test = train_test_split(x_axis, y_axis, random_state=0)
    
   # scaler = MinMaxScaler()
   # X_train = scaler.fit_transform(X_train)
   # X_test = scaler.fit_transform(X_test) 
 
    

    
    Linreg = Ridge(alpha=.01).fit(X_train, y_train)
    
    print('R-squared score (training): {:.3f}'
     .format(Linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}'
     .format(Linreg.score(X_test, y_test)))
    print('Linreg coefficients: :\n{}'
     .format(Linreg.coef_))
    print('Linreg interscept::\n{} '
     .format(Linreg.intercept_))
    print('MSE: %.2f'
     %mean_squared_error(y_test,Linreg.predict(X_test)))
    
    print('Variance score %.2f' %r2_score(y_test,Linreg.predict(X_test)))
    
    model = feature_selection.SelectKBest(score_func=feature_selection.f_regression,k=2)
    
    results = model.fit(X_train,y_train)
    
    print (results.scores_)
    print (results.pvalues_)
    
    plot_results(data,Linreg)
    
def plot_results(data,Linreg):
    
    data = data[data[:,0].argsort()]
    
    plt.scatter(data[:,0],data[:,2],color='black',label="Sales")
    
    plt.plot(data[:,0],Linreg.predict(data[:,[0,1]]),color="blue",label="Linear Regression")
  #  plt.plot(data[:,0],Linreg.predict(x_axis),color="blue",label="Linear Regression")
    
    plt.xlabel("Transactions")
    plt.ylabel("Sales")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()
    
   
    data = data[data[:,1].argsort()]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(data[:,0],data[:,1],Linreg.predict(data[:,[0,1]]),color='Red',label="Linear Regression")
    ax.scatter(data[:,0],data[:,1],data[:,2],c='blue',marker='o',label="Sales")
   
    ax.legend()

    plt.show()
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()
