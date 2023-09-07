from cProfile import label
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split

def ReplaceNullValues(dataset):
    #Replace Unknown values ( -999, -100, -1, 9999) with NaNs
    #List columns and its NaN values
    list_columns_minus_one_Nan = ['MonthlyIncome', 'external_score_1', 'external_score_2']
    list_columns_minus_999_Nan = ['repayment_amount']
    list_columns_minus_100_Nan = ['time_to_activation']
    list_columns_8000_Nan = ['external_score_2']
    list_columns_9999_Nan = ['external_score_2']
    
    #Replace Values
    dataset[list_columns_minus_one_Nan] = dataset[list_columns_minus_one_Nan].replace(-1, np.nan)
    dataset[list_columns_minus_999_Nan] = dataset[list_columns_minus_999_Nan].replace(-999, np.nan)
    dataset[list_columns_minus_100_Nan] = dataset[list_columns_minus_100_Nan].replace(-100, np.nan)
    dataset[list_columns_8000_Nan] = dataset[list_columns_8000_Nan].replace(8000, np.nan)
    dataset[list_columns_9999_Nan] = dataset[list_columns_9999_Nan].replace(9999, np.nan)
    
    return dataset

def EncodeColumn(dataset, column):
    #Encode the LivingStatus Variable
    dataset[column + '_Coded'] = pd.Categorical(dataset[column] ).codes
    return dataset

def totalTransactionsColumn(dataset):
    #Sum all columns of type "amount_transaction_typeX" to obtain the total amount spent 
    dataset['total_amount_transaction'] = 0
    for i in range(16):
        dataset['total_amount_transaction'] = dataset['total_amount_transaction'] + dataset[str('amount_transaction_type') + str(i+1)]
    return dataset

def oneHotEncode(dataset):
    #One-hot encode the LivingStatus and EmploymentStatus columns
    #one_hot_living = pd.get_dummies(dataset['LivingStatus']).drop(['Unknown'], axis = 1).rename(columns={'Other':'OtherLiving'})
    #one_hot_employment = pd.get_dummies(dataset['EmploymentStatus']).drop(['Unknown'], axis = 1).rename(columns={'Other':'OtherEmployment'})

    one_hot_living = pd.get_dummies(dataset['LivingStatus']).rename(columns={'Other':'OtherLiving', 'Unknown' : 'UnknownLiving'})
    one_hot_employment = pd.get_dummies(dataset['EmploymentStatus']).rename(columns={'Other':'OtherEmployment', 'Unknown' : 'UnknownEmployment'})

    dataset = dataset.join(one_hot_living)
    dataset = dataset.join(one_hot_employment)
    return dataset

def fillNaMean(dataset):
    #Fill NaN values with mean value from that feature
    dataset = dataset.fillna(dataset.mean(numeric_only = True))
    return dataset

"""
def fillNaIncome(dataset):
    average_income_employment = dataset[['MonthlyIncome', 'EmploymentStatus']].rename(columns = {'MonthlyIncome': 'AverageIncome'}).groupby(['EmploymentStatus'])['AverageIncome'].mean()
    data = pd.merge(dataset, average_income_employment, on = 'EmploymentStatus')
    data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['AverageIncome'])
    data = data.drop(labels = ['AverageIncome'], axis = 1)
    
    return data
"""

