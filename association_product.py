import pandas as pd
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def main():
   
    Create_item_list()


def Create_item_list():
    
    current_file = os.path.abspath(os.path.dirname(__file__)) #older/folder2/scripts_folder

#csv_filename
    csv_filename = os.path.join(current_file, '../Data/D11-02/D01')
    Data = pd.read_csv(csv_filename,sep=';')

    
    basket =Data.groupby(['cust_id','product_id'])['amount'].sum().unstack().reset_index().fillna(0).set_index('cust_id')
    
    basket_sets = basket.applymap(encode_units)
    
    frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
    
    print(frequent_itemsets)
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print(rules.head())
    
    
    #product_freq = Data[['product_id']].groupby('product_id').count()
    
    
    
    
    
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

   
    
    
if __name__ == '__main__':
    main()