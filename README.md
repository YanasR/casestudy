# casestudy

Download the code and run Casestudy.py --it produce 6 pyplots
The program reads data to pandas dataframe plot total sales by date,total transactions by date, sales per week of the day
it then fit a Ridge regression with amount of item sold on a day and number of transaction on a day as input feature
and predict total sale amount of that day.It also plot the precicted value,actual values for the input features

Run predicting_feature_creation.py separately-it uses seasonal modeling technique,the input features are index of the data(number
from 1 to total number of records in accending order of date),sin(2*pi*t/7) ,seven is selected since.The time series plot
clearly indicate a cycle for every 7 days.More features are created using polynomial regression before fitting. 
