# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Automated feature generation

#%% import libraries
import pandas as pd
import featuretools as ft

#%% Remove any limit on the number of columns to display 
pd.options.display.max_columns = None 

#%% Remove any limit on the number of rows to display
pd.options.display.max_rows = None

#%% Display the list of primitives
print(ft.list_primitives())

#%% create sample data
Customers = pd.DataFrame({ \
    'C_ID': ['C1', 'C2'], \
    'Name': ['Martin', 'Julia'], \
    'Creation_date': ['2018-08-15', '2020-05-05']}, \
        columns = ['C_ID','Name','Creation_date'])
Orders = pd.DataFrame({ \
    'Ord_ID': ['1', '2', '3', '4', '5'], \
    'C_ID': ['C1', 'C2', 'C1', 'C1','C2']}, \
        columns = ['Ord_ID','C_ID'])
Payments = pd.DataFrame({ \
    'Ord_ID':['1', '5', '3', '4', '2'], \
    'Price':[500, 200, 300, 100, 900]}, \
        columns = ['Ord_ID', 'Price'])

#%% create 'customer' entitysets
es = ft.EntitySet(id = 'Customers')
es = es.entity_from_dataframe( \
    entity_id = 'Customers', \
    dataframe = Customers, \
    index = 'C_ID', time_index = 'Creation_date')

#%% create orders entityset
es = es.entity_from_dataframe( \
    entity_id = 'Orders', \
    dataframe = Orders, \
    index = 'Ord_ID')

#%% create payments entityset
es = es.entity_from_dataframe( \
    entity_id = 'Payments', \
    dataframe = Payments, 
    make_index = True,
    index = 'P_ID')

#%%
# Define the relationship between the parent 'Customers' 
# and the child 'Orders' linked together by 'C_ID'
r_Cust_Ord = ft.Relationship( \
    es['Customers']['C_ID'], \
    es['Orders']['C_ID'])

#%% Add the relationship to the entity set
es = es.add_relationship(r_Cust_Ord)

#%% define relationship between 'Orders' 
# and 'Payments'
r_Orders_Payments = ft.Relationship( \
    es['Orders']['Ord_ID'], \
    es['Payments']['Ord_ID'])

#%% Add the relationship to the entity set
es = es.add_relationship(r_Orders_Payments)

#%% show entityset
es
# console output:
# Entityset: Customers
#   Entities:
#     Customers [Rows: 2, Columns: 3]
#     Orders [Rows: 5, Columns: 2]
#     Payments [Rows: 5, Columns: 3]
#   Relationships:
#     Orders.C_ID -> Customers.C_ID
#     Payments.Ord_ID -> Orders.Ord_ID

#%% show aggregation primitives
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 160
primitives[primitives['type']=="aggregation"].\
    head(15)

#%% show transformation primitives
primitives[primitives['type']=="transform"].\
    head(15)

#%% generate features
features, feature_names = ft.dfs( \
    entityset=es, \
    target_entity='Customers', \
    agg_primitives=['sum'], \
    trans_primitives=['year'])

features
# console output:
# 	    Name	SUM(Payments.Price)	YEAR(Creation_date)
# C_ID			
# C1	Martin	900	                2018
# C2	Julia	1100	            2020

#%% generate features
feats, feat_names = ft.dfs( \
    entityset=es, \
    target_entity='Customers', \
    max_depth = 2)

