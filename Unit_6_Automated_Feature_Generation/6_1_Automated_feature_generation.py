# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Automated feature generation

#%% import libraries
import pandas as pd
import featuretools as ft

#%% remove any limit on the number of columns to display 
pd.options.display.max_columns = None 

#%% remove any limit on the number of rows to display
pd.options.display.max_rows = None

#%% display the list of primitives
print(ft.list_primitives())

#%% create sample data
Customers = pd.DataFrame({
    'C_ID': ['C1', 'C2'],
    'Name': ['Martin', 'Julia'],
    'Creation_date': ['2018-08-15', '2020-05-05']},
    columns = ['C_ID', 'Name', 'Creation_date'])
Orders = pd.DataFrame({
    'Ord_ID': ['1', '2', '3', '4', '5'],
    'C_ID': ['C1', 'C2', 'C1', 'C1','C2']},
    columns = ['Ord_ID','C_ID'])
Payments = pd.DataFrame({
    'Ord_ID':['1', '5', '3', '4', '2'],
    'Price':[500, 200, 300, 100, 900]},
    columns = ['Ord_ID', 'Price'])

#%% create 'customer' entitysets
es = ft.EntitySet(id = 'Retail')
es = es.add_dataframe(
    dataframe_name = 'Customers',
    dataframe = Customers,
    index = 'C_ID', time_index = 'Creation_date')

#%% create orders entityset
es = es.add_dataframe(
    dataframe_name = 'Orders',
    dataframe = Orders,
    index = 'Ord_ID')

#%% create payments entityset
es = es.add_dataframe(
    dataframe_name = 'Payments',
    dataframe = Payments,
    make_index = True,
    index = 'P_ID')

#%% define the relationship between the parent 'Customers' 
# and the child 'Orders' linked together by 'C_ID'
es = es.add_relationship('Customers', 'C_ID', 'Orders', 'C_ID')

#%% define relationship between 'Orders' 
# and 'Payments'
es = es.add_relationship('Orders', 'Ord_ID', 'Payments', 'Ord_ID')

#%% show entityset
print(es)

# console output:
# Entityset: Retail
#   DataFrames:
#     Customers [Rows: 2, Columns: 3]
#     Orders [Rows: 5, Columns: 2]
#     Payments [Rows: 5, Columns: 3]
#   Relationships:
#     Orders.C_ID -> Customers.C_ID
#     Payments.Ord_ID -> Orders.Ord_ID

#%% show aggregation primitives
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 160
print(primitives[primitives['type']=="aggregation"])

#%% show transformation primitives
print(primitives[primitives['type']=="transform"])

#%% generate features
feats, feat_names = ft.dfs(entityset = es,
                           target_dataframe_name = 'Customers',
                           agg_primitives = ['sum'],
                           trans_primitives = ['year'])

print(feats)
# console output:
#       SUM(Payments.Price) YEAR(Creation_date)
# C_ID                                         
# C1                  900.0                2018
# C2                 1100.0                2020

#%% generate features
feats, feat_names = ft.dfs(entityset = es,
                           target_dataframe_name = 'Customers',
                           max_depth = 2)
