import pandas

from kagglehub import dataset_download

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


dataset_path_string = dataset_download(
    'marusagar/bank-transaction-fraud-detection')
dataset_path = Path(dataset_path_string) / \
    'Bank_Transaction_Fraud_Detection.csv'
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

data_frame = pandas.read_csv(dataset_path)
data_frame['Transaction_Hour'] = pandas.to_datetime(
    data_frame['Transaction_Time'], format='%H:%M:%S').dt.hour
data_frame.drop(columns=['Transaction_Time'], inplace=True)

y = data_frame.Is_Fraud
features = ['Gender', 'State', 'City', 'Bank_Branch', 'Account_Type',
            'Transaction_ID', 'Transaction_Date', 'Transaction_Type',
            'Merchant_Category', 'Transaction_Device', 'Device_Type',
            'Customer_Contact', 'Transaction_Location', 'Transaction_Currency',
            'Transaction_Description', 'Customer_Email', 'Age', 'Transaction_Amount',
            'Account_Balance', 'Transaction_Hour']

data_frame_for_encoding = data_frame[features]

final_data_frame = encoder.fit_transform(data_frame_for_encoding)

training_model = RandomForestRegressor(random_state=1)
training_model.fit(final_data_frame, y)
