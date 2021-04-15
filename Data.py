from datetime import datetime
import pandas as pd
import numpy as np
import math
import time
import os
from Seq2Img import get_image_with_price, image_loader, tensor_to_PIL
import matplotlib.pyplot as plt
import torch


# 用于投资组合的数据集
class DataMatrices:
    def __init__(self, batch_size=50, window_size=20, coin_number=10, feature_number=4, test_portion=0.15,
                 portion_reversed=False, is_permed=False, buffer_bias_ratio=0):
        self.feature_number = feature_number
        self.__batch_size = batch_size
        self._window_size = window_size
        self.__coin_no = coin_number
        self.test_portion = test_portion
        self.portion_reversed = portion_reversed
        self.__is_permed = is_permed
        self.__features = get_type_list(self.feature_number)
        self.__history_manager = HistoryManager(coin_number=self.__coin_no)
        # [feature, coin, date]
        self.__global_data = self.__history_manager.get_global_panel(feature_list=self.__features)
        # major: coin, minor: date
        self.__PVM = pd.DataFrame(np.ones((len(list(self.__global_data.minor_axis)), self.__coin_no))/self.__coin_no,
                                  index=list(self.__global_data.minor_axis),
                                  columns=list(self.__global_data.major_axis))
        self._num_periods = len(self.__global_data.minor_axis)
        self.__divide_data(test_portion, portion_reversed)

        end_index = self._train_ind[-1]
        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                            end_index=end_index,
                                            sample_bias=buffer_bias_ratio,
                                            batch_size=self.__batch_size,
                                            coin_number=self.__coin_no,
                                            is_permed=self.__is_permed)

        print("the number of training examples is %s"
              ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        print("the training set is from %s to %s" %
              (min(self._train_ind), max(self._train_ind)))
        print("the test set is from %s to %s" %
              (min(self._test_ind), max(self._test_ind)))

    def get_test_set(self):
        return self.__pack_samples(self.test_indices())

    def next_batch(self):

        batch = self.__pack_samples(
            [exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs - 1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w

        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        # [batch, feature, coin, date]
        # feature: 'CLOSE', 'HIGH', 'LOW', 'OPEN'
        x = M[:, :, :, :-1]
        # related price [batch, feature, coin]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        return {"data": x, "rp": y, "last_w": last_w, "setw": setw}

    def get_submatrix(self, ind):
        return self.__global_data.values[:, :, ind:ind+self._window_size+1]

    def __divide_data(self, test_portion, portion_reversed):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        # train and test set reverse
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[:-(self._window_size + 1)]
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self.test_indices())

    def test_indices(self):
        return self._test_ind[:-(self._window_size+1)]


# 用于趋势预测的数据集
class DataMatricesForTrend:
    def __init__(self, batch_size=50, window_size=20, coin_number=10, feature_number=5, test_portion=0.15,
                 trend_size=20, portion_reversed=False, is_permed=False, buffer_bias_ratio=0):
        self.feature_number = feature_number
        self.__batch_size = batch_size
        self._window_size = window_size
        self._trend_size = trend_size
        self.__coin_no = coin_number
        self.test_portion = test_portion
        self.portion_reversed = portion_reversed
        self.__is_permed = is_permed
        self.__features = get_type_list(self.feature_number)
        self.__history_manager = HistoryManager(coin_number=self.__coin_no)
        # [feature, coin, date]
        self.__global_data = self.__history_manager.get_global_panel(feature_list=self.__features)
        # major: coin, minor: date
        # self.__PVM = pd.DataFrame(np.ones((len(list(self.__global_data.minor_axis)), self.__coin_no))/self.__coin_no,
        #                          index=list(self.__global_data.minor_axis),
        #                          columns=list(self.__global_data.major_axis))
        self._num_periods = len(self.__global_data.minor_axis)
        self.__divide_data(test_portion, portion_reversed)

        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                            end_index=self._train_ind[-1],
                                            sample_bias=buffer_bias_ratio,
                                            batch_size=self.__batch_size,
                                            coin_number=self.__coin_no,
                                            is_permed=self.__is_permed)

        print("the number of training examples is %s"
              ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        print("the training set is from %s to %s" %
              (min(self._train_ind), max(self._train_ind)))
        print("the test set is from %s to %s" %
              (min(self._test_ind), max(self._test_ind)))

    def get_test_set(self):
        return self.__pack_samples(self.test_indices())

    def next_batch(self):

        batch = self.__pack_samples(
            [exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)

        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        # [batch, feature, coin, date]
        # feature: 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'
        x = M[:, :-1, :, self._window_size:-self._trend_size]
        volume = M[:, -1:, :, self._window_size:-self._trend_size]

        MA = [np.mean(M[:, :1, :, a:a+self._window_size], axis=3) for a in range(self._window_size)]
        MA = np.array(MA)
        MA = MA.transpose((1, 2, 3, 0))

        x = np.concatenate((x, MA), axis=1)
        x = np.concatenate((x, volume), axis=1)
        x = x.transpose((0, 2, 3, 1))

        data = []

        for i in range(x.shape[0]):
            item = []
            for j in range(x.shape[1]):
                data_item = x[i][j]
                image = get_image_with_price(data_item)
                tensor_value = image_loader(image)
                list_value = tensor_value.numpy().tolist()
                item.append(list_value)
            data.append(item)
        data = torch.Tensor(data)

        # trend [batch, coin]
        z = M[:, 0, :, -1] / M[:, 0, :, -(self._trend_size+1)]
        # Ture False
        z = z > 1
        # 1 0
        z = z + 0
        return {"data": data, "trend": z}

    def get_submatrix(self, ind):
        return self.__global_data.values[:, :, ind - self._window_size:ind+self._window_size + self._trend_size]

    def __divide_data(self, test_portion, portion_reversed):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        # train and test set reverse
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[self._window_size:-(self._window_size + self._trend_size)]
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self.test_indices())

    def test_indices(self):
        return self._test_ind[self._window_size:-(self._window_size + self._trend_size)]


class HistoryManager:
    def __init__(self, coin_number):
        self._coin_number = coin_number

    def get_global_panel(self, feature_list=['CLOSE', 'HIGH', 'LOW', 'OPEN']):
        file_list = [f for f in os.listdir('./database') if os.path.isfile(os.path.join('./database', f))]
        coin_list = [i[:-4] for i in file_list]
        date_list = []
        data_np = []

        if os.path.exists('./database/data_new.pkl'):
            panel = pd.read_pickle("./database/data_new.pkl")
        else:
            for i in range(len(file_list)):
                item_df = pd.read_csv('./database/' + file_list[i])
                item_np = item_df[feature_list].values
                data_np.append(item_np)
                if len(date_list) == 0:
                    date_list = item_df[item_df.columns[0]]
            # print(np.array(data_np).shape)

            panel = pd.Panel(items=feature_list, major_axis=coin_list,
                             minor_axis=date_list, dtype=np.float32)

            for i in range(len(coin_list)):
                for j in range(len(feature_list)):
                    for k in range(len(date_list)):
                        panel.loc[feature_list[j], coin_list[i], date_list[k]] = data_np[i][k][j]

            f = open('./database/data_new.pkl', 'wb')
            panel.to_pickle(f)
            f.close
        # [features, coins, dates]
        return panel


def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ['CLOSE']
    elif feature_number == 2:
        type_list = ['CLOSE', 'OPEN']
    elif feature_number == 3:
        type_list = ['CLOSE', 'HIGH', 'LOW']
    elif feature_number == 4:
        type_list = ['CLOSE', 'HIGH', 'LOW', 'OPEN']
    elif feature_number == 5:
        type_list = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list


class ReplayBuffer:
    def __init__(self, start_index, end_index, batch_size, is_permed, coin_number, sample_bias=1.0):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self.__coin_number = coin_number
        self.__experiences = [Experience(i)
                              for i in range(start_index, end_index)]
        self.__is_permed = is_permed
        # NOTE: in order to achieve the previous w feature
        self.__batch_size = batch_size
        self.__sample_bias = sample_bias
        print("buffer_bias is %f" % sample_bias)

    def append_experience(self, state_index):
        self.__experiences.append(Experience(state_index))
        print("a new experience, indexed by %d, was appended" % state_index)

    def __sample(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        # TODO: deal with the case when bias is 0
        ran = np.random.geometric(bias)
        while ran > end - start or ran < start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
        if self.__is_permed:
            for i in range(self.__batch_size):
                # print(i)
                batch.append(self.__experiences[self.__sample(self.__experiences[0].state_index,
                                                              self.__experiences[-1].state_index,
                                                              self.__sample_bias)])
        else:
            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,
                                        self.__sample_bias)
            batch = self.__experiences[batch_start:batch_start +
                                       self.__batch_size]
        return batch


class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)


if __name__ == '__main__':
    batch_size = 32
    window_size = 20
    trend_size = 20
    coin_number = 10
    DM = DataMatricesForTrend(batch_size=batch_size, window_size=window_size, coin_number=coin_number, feature_number=5,
                              test_portion=0.15,trend_size=trend_size, portion_reversed=False, is_permed=True,
                              buffer_bias_ratio=5e-5)

    nextbatch = DM.next_batch()
    data = nextbatch['data']
    result = nextbatch['trend']

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            image = tensor_to_PIL(data[i][j])
            plt.imshow(image)
            plt.show()
















