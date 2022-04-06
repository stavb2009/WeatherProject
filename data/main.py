import math
from typing import Tuple
from datetime import datetime
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import dataLoader
import model as model_l
import pandas as pd
import numpy as np

w = 2
train_constructor = dataLoader.WindowSlider(window_size=w)

dataset_location = 'dead_see_weather_dates_edited.csv'
df = pd.read_csv(dataset_location)

columns_to_num = ['Wsp_WS4_Avg', 'Wdr_WS4_Avg', 'Wdr_WS4_Std', 'Wsp_WS4_Max',
                  'Lv_PA36_Avg', 'Lv_PA36_Max', 'Lv_PA36_Min', 'Lv_PA36_Std']

df3 = train_constructor.dataset_prep(dataset_location, name_of_dates_column='TIMESTAMP',
                                     columns_to_floats=columns_to_num)

deltaT = np.array([((df3.index[i + 1] - df3.index[i]).value) / 1e10 for i in range(len(df3) - 1)])
deltaT = np.concatenate((np.array([0]), deltaT))

df3.insert(2, '∆T', deltaT)


data_set = df3["2018-12-12":"2018-12-25"]
data_set.drop(columns=['TIMESTAMP', 'minute_of_day', 'day', 'month', 'year'], axis=1, inplace=True)

data_tmp = data_set[['Wdr_WS4_Avg', 'Wdr_WS4_Std']]
#   train_windows = train_constructor.collect_windows(data_set, save_mode=True, window_size=w, step=w)
#   train_plus_train_windows = train_constructor.Add_predictions(data_tmp, prediction_length=2)

# selected_columns = ['Wsp_WS4_Avg(1)',
#                    'Wsp_WS4_Avg(2)','∆T(1)', '∆T(2)', 'Wdr_WS4_Avg(1)', 'Wdr_WS4_Avg(2)']

selected_columns = ['Wsp_WS4_Avg', 'Wdr_WS4_Std', 'Wsp_WS4_Max']

train = torch.from_numpy((data_set[selected_columns][10:20]).values).float()
valid = torch.from_numpy((data_set[selected_columns][20:30]).values).float()
test = torch.from_numpy((data_set[selected_columns][30:40]).values).float()


######################################################################

# until here its only the data

######################################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """

    #TODO:  In real run we have to change the line under to fit a length of half day
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


tmp_dt = {
        "X": [3, 5, -2, 2],
        "Y": [7, -1, -1, -9]
    }
tmp_forcast = {
        "X": [1, 2, 3],
        "Y": [-1, -2, -3]
    }


dt = pd.DataFrame(tmp_dt)
forcast_dt = pd.DataFrame(tmp_forcast)

dt = pd.DataFrame(tmp_dt)
forcast_dt = pd.DataFrame(tmp_forcast)
print(dt)

dt = dt.transpose()
forcast_dt = forcast_dt.transpose()
print("forcast dt: \n",forcast_dt)
print("\n dt: \n",dt)


ntokens = 2  # len(selected_columns)  # size of data that we put inside # the number of rows in the input
d_model = ntokens  # embedding dimension # but in our case it can be an arbitrary
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 #int(len(selected_columns)/w)  # number of heads in nn.MultiheadAttention # I supose that it shold be the number of variables that we have
dropout = 0.2  # dropout probability
model = model_l.TransformerModel(ntokens, d_model, nhead, d_hid, nlayers, dropout).to(device)
bptt = ntokens


####################################

#     until here everything is ok

####################################

src_mask = model_l.generate_square_subsequent_mask(bptt).to(device)
data, targets = get_batch(train, 3)
src_mask = src_mask[:bptt, :bptt]
criterion = nn.MSELoss()

output = model(train[2], src_mask)
loss = criterion(output.view(-1, ntokens), targets)


# why is the shape of the output is [10, 1, 6] ?  I should run the exampel of the nlp transformer on GOGGLE_COLAB