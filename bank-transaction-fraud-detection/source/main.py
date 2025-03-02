import pandas

from kagglehub import dataset_download

from pathlib import Path

from fastai.tabular.all import (
    TabularDataLoaders,
    tabular_learner,
    accuracy,
    Categorify,
    FillMissing,
    Normalize
)


dataset_path_string = dataset_download(
    'marusagar/bank-transaction-fraud-detection')
dataset_path = Path(dataset_path_string) / \
    'Bank_Transaction_Fraud_Detection.csv'

data_frame = pandas.read_csv(dataset_path)
data_frame['Transaction_Hour'] = pandas.to_datetime(
    data_frame['Transaction_Time'], format='%H:%M:%S').dt.hour
data_frame.drop(columns=['Transaction_Time'], inplace=True)

data_block = TabularDataLoaders.from_df(
    data_frame,
    y_names='Is_Fraud',
    cat_names=['Gender', 'State', 'City', 'Bank_Branch', 'Account_Type',
               'Transaction_ID', 'Transaction_Date', 'Transaction_Type',
               'Merchant_Category', 'Transaction_Device', 'Device_Type',
               'Customer_Contact', 'Transaction_Location', 'Transaction_Currency',
               'Transaction_Description', 'Customer_Email'],
    cont_names=['Age', 'Transaction_Amount',
                'Account_Balance', 'Transaction_Hour'],
    procs=[Categorify, FillMissing, Normalize]
)


learner = tabular_learner(data_block, metrics=accuracy)
learner.fit_one_cycle(3)
