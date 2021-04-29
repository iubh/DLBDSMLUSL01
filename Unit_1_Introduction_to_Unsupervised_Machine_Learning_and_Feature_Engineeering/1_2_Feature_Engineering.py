# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature Engineering

#%% import libraries
import numpy as np
import pandas as pd
import datetime

#%% create sample data
Student_R = { \
    'Student_ID':['S1', 'S2', 'S3'], \
    'Birth_date': [datetime.date(1996,7,14), \
                   datetime.date(1997,8,22), \
                   datetime.date(1998,5,11)]}

Student_R = pd.DataFrame(Student_R, \
    columns = ['Student_ID','Birth_date'])

Courses = { \
    'Student_ID':['S1', 'S2', 'S3', 'S1', 'S2', 'S3'], \
    'Grades':[18, 11, 12, 15, 19, 10]}

Courses = pd.DataFrame (Courses, \
    columns = ['Student_ID', 'Grades'])

#%% extracting the year from the birth date
Student_R['year'] = pd.DatetimeIndex(Student_R['Birth_date']).year
print(Student_R.head())

# console output:
#   Student_ID  Birth_date  year
# 0         S1  1996-07-14  1996
# 1         S2  1997-08-22  1997
# 2         S3  1998-05-11  1998

#%% creation of features by aggregation of grouped values
goper = Courses.groupby('Student_ID')['Grades'].\
    agg(['mean','max','min'])

# rename columns
goper.columns = ['mean_grade','max_grade','min_grade']
print(goper.head())

# console output:
#             mean_grade  max_grade  min_grade
# Student_ID                                  
# S1                16.5         18         15
# S2                15.0         19         11
# S3                11.0         12         10


#%% merge with the Student_R dataframe
R = Student_R.merge(goper, left_on = 'Student_ID', \
    right_index=True, how = 'left'). \
        head()

# show the dataframe
R

#   Student_ID  Birth_date  year  mean_grade  max_grade  min_grade
# 0         S1  1996-07-14  1996        16.5         18         15
# 1         S2  1997-08-22  1997        15.0         19         11
# 2         S3  1998-05-11  1998        11.0         12         10
