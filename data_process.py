import numpy as np
import reader

class DataSet(object):

    def __init__(self, datas):
        # initialization

        self._datas = datas
        self._location = 0
        self._numbers = len(datas)

    def next_batch(self, batch_size):
        #get the batch data for trainning
        start = self._location
        self._location += batch_size
        if self._location > self._numbers:
            np.random.shuffle(self._datas)
            start = 0
            self._location = batch_size
        end = self._location

        this_batch = self._datas[start:end]
        return reader.get_data(this_batch)

    @property
    def datas(self):
        return self._datas

    @property
    def location(self):
        return self._location

    @property
    def numbers(self):
        return self._numbers

#return the Class of datas
def do_it(datas):
    Data = DataSet(datas)
    return Data
