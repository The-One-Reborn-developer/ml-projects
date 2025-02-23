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
dataset_path = Path(dataset_path_string) / 'Bank_Transaction_Fraud_Detection.csv'

data_block = TabularDataLoaders.from_csv(
    dataset_path,
    y_names='Is_Fraud',
    cat_names=['Customer_ID', 'Gender', 'State', 'City', 'Bank_Branch',
               'Account_Type', 'Transaction_ID', 'Transaction_Date', 'Transaction_Time',
               'Merchant_ID', 'Transaction_Type', 'Merchant_Category',
               'Transaction_Device', 'Device_Type', 'Customer_Contact', 'Transaction_Location',
               'Transaction_Currency', 'Transaction_Description', 'Customer_Email'],
    cont_names=['Age', 'Transaction_Amount', 'Account_Balance'],
    procs=[Categorify, FillMissing, Normalize]
)


learner = tabular_learner(data_block, metrics=accuracy)
learner.fit_one_cycle(3)
