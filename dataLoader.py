import numpy as np
import pandas as pd
from datetime import datetime
import re
import torch
from torch import nn, Tensor
import time
import xlsxwriter

"""
I took all of that from this site:
https://towardsdatascience.com/ml-approaches-for-time-series-4d44722e48fe
"""


class Data(object):
    def convert_panda_to_tensors(panda: pd.DataFrame, numOfParameters=2) -> Tensor:
        """
        Args:
        This function take data with prediction and positional encoding and convert it
        to tensor, when it's first dimention (dim=0) would be the sample umber
        :return:
        """

        number_of_samples = int(panda.shape[0] / (numOfParameters + 1))
        number_of_measurements_in_sample = panda.shape[1] - 1
        # x_and_y = 2

        tensor_data = torch.zeros((number_of_samples, (numOfParameters + 1), number_of_measurements_in_sample))
        tensor_day_in_year = torch.zeros(number_of_samples)
        df_time = panda.loc[panda[panda.columns[0]] == ('TIMESTAMP' or 'time')]
        df_without_time = panda.loc[panda[panda.columns[0]] != ('TIMESTAMP' or 'time')]

        idx_df = 0
        idx_tens = 0
        while idx_tens < number_of_samples:
            tensor_day_in_year[idx_tens] = np.cos(
                2 * np.pi * (pd.to_datetime(df_time.iloc[idx_tens, 1]).dayofyear / 365))
            if np.cos(2 * np.pi * (pd.to_datetime(df_time.iloc[idx_tens, 1]).dayofyear / 365)) > 1:
                print("stop")
            tensor_data[idx_tens, :numOfParameters] = torch.from_numpy(
                (df_without_time[panda.columns[1::]][idx_df:idx_df + numOfParameters]).values.astype(np.float64))
            idx_df += numOfParameters
            idx_tens += 1

        tensor_day_in_year = tensor_day_in_year.repeat(tensor_data.shape[2], 1).transpose(0, 1)
        tensor_data[:, numOfParameters, :] = tensor_day_in_year
        # tensor_data = tensor_data.transpose(1, 2)
        tensor_data = tensor_data.reshape(
            (tensor_data.shape[0], 1, int(tensor_data.shape[1] * tensor_data.shape[2])))
        return tensor_data

    def batchify(trains: Tensor, result: Tensor, samps_in_batch: int = 1, shuffle=True) -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            train: Tensor , train data
            result: Tensor, the expected result

        Returns:
            returns a list which every position is (train, result)
        """
        data_train = trains
        data_res = result
        if shuffle:
            idx = torch.randperm(trains.shape[0])
            data_train = data_train[idx]
            data_res = data_res[idx]

        data_train = torch.split(data_train, samps_in_batch, dim=0)
        data_res = torch.split(data_res, samps_in_batch, dim=0)
        baches = [(data_train[i], data_res[i]) for i in range(len(data_res))]

        return baches

    def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Modified from:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        Args:
            dim1: int, for both src and tgt masking, this must be target sequence
                  length
            dim2: int, for src masking this must be encoder sequence length (i.e.
                  the length of the input sequence to the model),
                  and for tgt masking, this must be target sequence length
        Return:
            A Tensor of shape [dim1, dim2]
        """
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


class WindowSlider(object):

    def __init__(self, window_size=5, start_index=1):
        '''
        Window Slider object
        ====================
        w: window_size - number of time steps to look back
        o: offset between last reading and temperature
        s: response_size - number of time steps to predict
        l: length to slide - ((#observation - w)/s)+1
        start_index: tells from Which measurement We Start
        '''
        self.w = window_size
        self.s = 1
        self.l = 0
        self.names = []
        self.Data = pd.DataFrame()

    def re_init(self, arr):
        '''
        Helper function to initializate to 0 a vector
        '''
        arr = np.cumsum(arr)
        return arr - arr[0]

    def collect_windows(self, X, save_mode=False, window_size=5, step=1, start_index=0):
        '''
        Input: X is the input matrix, each column is a variable
        Returns: diferent mappings window-output
        '''
        self.names = []
        cols = len(list(X))
        N = len(X)

        self.w = window_size
        self.s = step
        self.l = int(np.floor((N - self.w) / self.s) + 1)

        if self.l < 0:
            print("Class: WindowSlider, def: collect_windows, error: Length od input data is too short for the window")
            return -1

        # Create the names of the variables in the window
        # Check first if we need to create that for the response itself

        self.names.append(X.index.name + "(Start time)")
        self.names.append(X.index.name + "(End time)")
        for j, col in enumerate(list(X)):

            for i in range(self.w):
                name = col + ('(%d)' % (i + start_index + 1))
                self.names.append(name)
        #
        # # Incorporate the timestamps where we want to predict
        # for k in range(self.s):
        #     name = '∆t' + ('(%d)' % (self.w + k + 1))
        #     self.names.append(name)

        df = pd.DataFrame(np.zeros(shape=(self.l, len(self.names))),
                          columns=self.names)

        # Populate by rows in the new dataframe
        for i in range(self.l):

            slices = np.array([X.index[i * self.w], X.index[i * self.w + self.w - 1]])

            # Flatten the lags of predictors
            for p in range(X.shape[1]):
                line = X.values[i * self.s:self.w + i * self.s, p]

                # Concatenate the lines in one slice
                slices = np.concatenate((slices, line))

                # Incorporate the timestamps where we want to predict

            # Incorporate the slice to the cake (df)

            if len(slices) == 12:
                pass

            df.iloc[i, :] = slices

        if save_mode:
            self.Data = df.copy(deep=True)
        return df

    def Add_predictions(self, predictions=None, prediction_length=5):

        W = self.w
        prediction_windows = self.collect_windows(predictions, window_size=prediction_length, save_mode=False,
                                                  step=prediction_length, start_index=self.w)
        self.Data.insert(len(self.Data.columns), 'Start_of_prediction', np.full(len(self.Data), np.nan),
                         allow_duplicates=True)
        self.Data.insert(len(self.Data.columns), 'End_of_prediction', np.full(len(self.Data), np.nan),
                         allow_duplicates=True)

        for i in predictions.columns:
            for j in range(prediction_length):
                self.Data.insert(loc=(self.Data.columns.get_loc(i + ('(%d)' % (W + j))) + 1),
                                 column=(i + ('(%d)' % (W + j + 1))), value=np.full(len(self.Data), np.nan),
                                 allow_duplicates=True)

        for index, date in enumerate(prediction_windows['Date_format(Start time)']):
            for row_num in range(len(self.Data)):
                if ((date - self.Data['Date_format(End time)'][row_num]).value == 600000000000):
                    self.Data.loc[row_num, 'Start_of_prediction': 'End_of_prediction'] = \
                        list(prediction_windows.loc[index, 'Date_format(Start time)': 'Date_format(End time)'])

                    for colum_name in predictions.columns:
                        start_column_name = (colum_name + ('(%d)' % (W + 1)))
                        end_column_name = (colum_name + ('(%d)' % (W + prediction_length)))
                        self.Data.loc[row_num, start_column_name: end_column_name] = \
                            list(prediction_windows.loc[index, start_column_name: end_column_name])

        # For the next use of the prediction function
        self.w = W + prediction_length

        return self.Data

    def dataset_prep(self, dataset_location=None, name_of_dates_column=None, columns_to_floats=None):
        if not dataset_location:
            print("No DataSet location have inserted")
            return -1

        df = pd.read_csv(dataset_location)

        dates = df[name_of_dates_column]
        list_times = [dates[i].split(' ') for i in range(2, len(dates))]
        days, hour_of_day = zip(*list_times)
        #

        hour_of_day_format = ['0' + sub if re.search("^.:", sub) else sub for sub in hour_of_day]
        minute_of_day = [(int(hour_of_day[i].split(':')[0])) * 60 + (int(hour_of_day[i].split(':')[1])) for i in
                         range(len(hour_of_day))]
        days2 = [days[i].split('/') for i in range(len(days))]
        month, day, year = zip(*days2)
        month_format = ['0' + sub if len(sub) < 2 else sub for sub in month]
        time_format = [
            datetime.fromisoformat(year[i] + '-' + month_format[i] + '-' + day[i] + ' ' + hour_of_day_format[i])
            for i in range(len(days))]
        day = [int(day[i]) for i in range(len(day))]
        month = [int(month[i]) for i in range(len(month))]
        year = [int(year[i]) for i in range(len(year))]

        df = df[2::].copy(deep=True)
        df["minute_of_day"] = minute_of_day
        df["day"] = day
        df["month"] = month
        df["year"] = year
        df["Date_format"] = time_format
        df.set_index('Date_format', inplace=True)
        for column in columns_to_floats:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        return df
