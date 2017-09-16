import pandas as pd
import os
import matplotlib.pyplot as plt
from _datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import feature_selection
import matplotlib.pyplot as py
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import math

def main():
   
    predicting_sinusoidal()


def predicting_sinusoidal():
    
    current_file = os.path.abspath(os.path.dirname(__file__)) #older/folder2/scripts_folder

#csv_filename
    csv_filename = os.path.join(current_file, '../Data/D11-02/D_combined')
    Data = pd.read_csv(csv_filename,sep=';')

    
    Data =Data[['trans_date_time','price']].groupby('trans_date_time',as_index=False)['price'].sum()
    
    
    
    
    Data['trans_date_time'] =  pd.to_datetime( Data['trans_date_time'])
    
    Data['trans_date_time'] = Data['trans_date_time'].dt.weekday_name
    
    
    
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    
    Data['trans_date_time'] = Data['trans_date_time'].apply(lambda x: days[x])
    
    x_axis1 =Data.index.tolist()
    
    y_axis =Data['price']
        
    x_axis1 = np.array(x_axis1)
    
    x_axis2 = np.sin(2*math.pi*x_axis1/7)
    x_axis3 = np.cos(2*math.pi*x_axis1/7)
    y_axis = np.array(y_axis)
    x_axis = np.column_stack((x_axis1,x_axis2,x_axis3))
    data = np.column_stack((x_axis,y_axis))
    
    X_train, X_test, y_train, y_test = train_test_split(x_axis, y_axis, random_state=0)
    poly = PolynomialFeatures(degree=5)
    X_F1_poly = poly.fit_transform(X_train)
    X_F2_poly = poly.fit_transform(X_test)
    Linreg = Ridge(alpha=.1).fit(X_F1_poly, y_train)
    
 #   Linreg = Ridge(alpha=1).fit(X_train, y_train)
    

    print('R-squared score (training): {:.3f}'
     .format(Linreg.score(X_F1_poly, y_train)))
    print('R-squared score (test): {:.3f}'
     .format(Linreg.score(X_F2_poly, y_test)))
    print('Linreg coefficients: :\n{}'
     .format(Linreg.coef_))
    print('Linreg interscept::\n{} '
     .format(Linreg.intercept_))
    print('MSE: %.2f'
     %mean_squared_error(y_test,Linreg.predict(X_F2_poly)))
    
    print('Variance score %.2f' %r2_score(y_test,Linreg.predict(X_F2_poly)))
    
    model = feature_selection.SelectKBest(score_func=feature_selection.f_regression,k=2)
    
    results = model.fit(X_F1_poly,y_train)
    
    print (results.scores_)
    print (results.pvalues_)
    plot_results(data,Linreg,poly)
    
     
    
def plot_results(data,Linreg,poly):
    
    data = data[data[:,0].argsort()]
    
    plt.plot(data[:,0],data[:,3],color='black',label="Sales")
    
    X_F1_poly = poly.fit_transform(data[:,[0,1,2]])
    plt.plot(data[:,0],Linreg.predict(X_F1_poly),color="blue",label="Linear Regression")
  #  plt.plot(data[:,0],Linreg.predict(x_axis),color="blue",label="Linear Regression")
    
    plt.xlabel("Transactions")
    plt.ylabel("Sales")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()
    
   
    
   
   
   
    
   
    
if __name__ == '__main__':
    main()