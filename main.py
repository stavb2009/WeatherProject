import math
import torch
from torch import nn, Tensor
import os
import dataLoader
from data import model as model_l
import pandas as pd
import xlsxwriter
import torch.optim as opt

import copy
import time

torch.set_printoptions(linewidth=120)
import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import pickle
import openpyxl
import numpy as np

# w = 2
# train_constructor = dataLoader.WindowSlider(window_size=w)
#
# dataset_location = 'dead_see_weather_dates_edited.csv'
# df = pd.read_csv(dataset_location)
#
# columns_to_num = ['Wsp_WS4_Avg', 'Wdr_WS4_Avg', 'Wdr_WS4_Std', 'Wsp_WS4_Max',
#                   'Lv_PA36_Avg', 'Lv_PA36_Max', 'Lv_PA36_Min', 'Lv_PA36_Std']
#
# df3 = train_constructor.dataset_prep(dataset_location, name_of_dates_column='TIMESTAMP',
#                                      columns_to_floats=columns_to_num)
#
# deltaT = np.array([((df3.index[i + 1] - df3.index[i]).value) / 1e10 for i in range(len(df3) - 1)])
# deltaT = np.concatenate((np.array([0]), deltaT))
#
# df3.insert(2, '∆T', deltaT)
#
#
# data_set = df3["2018-12-12":"2018-12-25"]
# data_set.drop(columns=['TIMESTAMP', 'minute_of_day', 'day', 'month', 'year'], axis=1, inplace=True)
#
# data_tmp = data_set[['Wdr_WS4_Avg', 'Wdr_WS4_Std']]
# #   train_windows = train_constructor.collect_windows(data_set, save_mode=True, window_size=w, step=w)
# #   train_plus_train_windows = train_constructor.Add_predictions(data_tmp, prediction_length=2)
#
# # selected_columns = ['Wsp_WS4_Avg(1)',
# #                    'Wsp_WS4_Avg(2)','∆T(1)', '∆T(2)', 'Wdr_WS4_Avg(1)', 'Wdr_WS4_Avg(2)']
#
# selected_columns = ['Wsp_WS4_Avg', 'Wdr_WS4_Std', 'Wsp_WS4_Max']
#
# train = torch.from_numpy((data_set[selected_columns][10:20]).values).float()
# valid = torch.from_numpy((data_set[selected_columns][20:30]).values).float()
# test = torch.from_numpy((data_set[selected_columns][30:40]).values).float()


######################################################################

# until here its only the data

######################################################################

# load the data and define the device
###################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder = 'data_for_24_4'
file_train = 'data_040822.csv'
file_val = 'val_040822.csv'
src_train = os.path.join(folder, file_train)
src_val = os.path.join(folder, file_val)
df_train = pd.read_csv(src_train)
df_val = pd.read_csv(src_val)
train_tensor_row = dataLoader.Data.convert_panda_to_tensors(df_train, numOfParameters=3)
val_tensor_row = dataLoader.Data.convert_panda_to_tensors(df_val, numOfParameters=3)


# here are all the parameters of the network:
######################################################################


def train(model: nn.Module, random_numbers) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 20
    start_time = time.time()
    src_mask = model_l.generate_square_subsequent_mask(num_batch).to(device)
    src_mask.to(device)
    num_batches = len(random_numbers)

    for batch, i in enumerate(random_numbers):

        # If we want to use real batches, we need to look at the original code again!
        data, targets = train_tuple[i][0], train_tuple[i][1]
        data = data.to(device)
        targets = targets.to(device)
        output = model(data, src_mask).to(device)
        # we want to use only the wind and direction and not the day:
        loss = criterion(output[:2, :, :targets.shape[2]], targets[:2, :, :])  
        writer.add_scalar('training loss', loss, batch)  # used to be global_step=1

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # make sure it ('=') doesn't hurt
        writer.add_scalar('grad_norm', grad_norm, batch)  # used to be global_step=1
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('total loss', total_loss, batch)
        if (batch + 1) % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            writer.add_scalar('curr loss', cur_loss, batch)
            # ppl = 1 #math.exp(cur_loss)
            # TODO: we dont need the ppl, it's only here for Stav's clear conscious
            print(f'| epoch {epoch:3d} | {batch + 1:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
                  f' averaged loss {cur_loss:5.2f}')
            total_loss = 0
            start_time = time.time()

    # for batch, i in enumerate(train_tuple):
    #
    #     # If we want to use real batches, we need to look at the original code again!
    #     data, targets = train_tuple[random_numbers[batch]][0], train_tuple[random_numbers[batch]][0][1]
    #     output = model(data, src_mask)
    #     loss = criterion(output[:,:,:targets.shape[2]], targets)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #     optimizer.step()
    #
    #     total_loss += loss.item()
    #     if batch % log_interval == 0 and batch > 0:
    #         lr = scheduler.get_last_lr()[0]
    #         ms_per_batch = (time.time() - start_time) * 1000 / log_interval
    #         cur_loss = total_loss / log_interval
    #         ppl = math.exp(cur_loss)
    #         print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
    #               f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
    #               f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
    #         total_loss = 0
    #         start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = model_l.generate_square_subsequent_mask(num_batch).to(device)
    with torch.no_grad():
        for i in eval_data:
            data, targets = i[0].to(device), i[1].to(device)
            batch_size = num_batch
            output = model(data, src_mask)

            total_loss += batch_size * criterion(output[:, :, :targets.shape[2]], targets).item()
    return total_loss / (len(eval_data))


if __name__ == '__main__':
    epochs_list = range(10, 11, 5)
    nheads = [2, 8]  # int(len(selected_columns)/w)  # number of heads in nn.MultiheadAttention # I supose that it shold be the number of variables that we have
    # TODO: we need to see how many heads we need
    lrs = np.geomspace(1e-5, 50, num=8)  # learning rates
    epoch_sizes = range(40, 41, 10)
    num_of_batches = range(3, 4)
    d_hids = range(200, 206, 40)  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout = 0.2  # dropout probability
    best_val_loss = float('inf')
    best_model = None

    for epochs in epochs_list:
        for lr in lrs:
            for epoch_size in epoch_sizes:
                for num_batch in num_of_batches:
                    for nhead in nheads:
                        for d_hid in d_hids:
                            # writer_comment = "epochs = " + str(epochs) + f' lr ={lr:.} ' + str( lr) + " epoch_size
                            # = " + str( epoch_size) + ' num_batch = ' + str(num_batch) + ' nhead= ' + str(nhead) + '
                            # d_hids= ' \ + str(d_hid)
                            writer_comment = f'epochs = {epochs} ||  lr ={lr:1.6f} ||  epoch_size = {epoch_size} || ' \
                                             f' num_batch = {num_batch} || nhead = {nhead} || d_hids = {d_hid}'
                            print(writer_comment)
                            writer = SummaryWriter(comment=writer_comment)
                            ###
                            ###
                            train_tuple = dataLoader.Data.batchify(train_tensor_row, val_tensor_row,
                                                                   samps_in_batch=num_batch)  # changed
                            # Let's play with it a bit
                            ntokens = train_tuple[0][0].shape[2]  # len(selected_columns)
                            # size of data that we put inside # the number of
                                                                              # columns in the input
                            d_model = train_tuple[0][0].shape[2]  # embedding dimension
                                                                # but in our case it can be an arbitrary
                            model = model_l.TransformerModel(ntokens, d_model, nhead, d_hid, nlayers, dropout).to(
                                device)
                            criterion = nn.MSELoss()
                            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                            # TODO:we want to talk about it with Ayal
                            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
                            train_data = train_tensor_row

                            for epoch in range(1, epochs + 1):
                                epoch_start_time = time.time()
                                random_indexes = torch.arange(len(train_tuple))#torch.squeeze(torch.randint(0, len(train_tuple) - 1, (1, epoch_size)))
                                train(model, random_indexes)
                                #writer.add_graph(model)
                                val_loss = evaluate(model, train_tuple)
                                writer.add_scalar('val_loss', val_loss, epoch)
                                writer.add_histogram("weights decoder data", model.decoder.weight.data)
                                writer.add_histogram("weights decoder T", model.decoder.weight.T)
                                writer.add_histogram("weights decoder grad", model.decoder.weight.grad)
                                writer.add_scalar("weight decoder grad", torch.norm(model.decoder.weight.grad))
                                writer.add_histogram("bias  decoder", model.decoder.bias.data)

                                writer.add_histogram("weights encoder data", model.encoder.weight.data)
                                writer.add_histogram("weights encoder T", model.encoder.weight.T)
                                writer.add_histogram("weights encoder grad", model.encoder.weight.grad)
                                writer.add_scalar("weight encoder grad", torch.norm(model.encoder.weight.grad))
                                writer.add_histogram("bias encoder", model.encoder.bias.data)



                                # val_ppl = 1  # math.exp(val_loss)
                                elapsed = time.time() - epoch_start_time
                                #writer.flush()  # should it be here?
                                print('-' * 89)
                                print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                                      f'valid loss {val_loss:5.2f}')
                                print('-' * 89)

                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    best_model = copy.deepcopy(model)
                                    torch.save(best_model.state_dict(),
                                               os.path.join(os.getcwd(), 'data/model_trained.pt'))
                                scheduler.step()

                            writer.flush()
                            writer.close()


