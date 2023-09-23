import math
from typing import Union
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import mutual_info_score
class discreteEntropy :
    def __init__(self, message : str, word_length : int = None) :
        self._message = message
        self._word_length = word_length
    @property
    def word_length(self):
        return self._word_length
    @word_length.setter
    def word_length(self, value):
        self._word_length = value
    def shannon(self) -> float :
        exr = {}
        entropy = 0
        for i in self._message:
            try:
                exr[i] += 1
            except KeyError:
                exr[i] = 1
        textlen = len(self._message)
        for value in exr.values():
            freq = 1 * value / textlen
            entropy += freq * math.log(freq) / math.log(2)
        entropy *= -1
        return entropy
    def lempel_ziv(self) -> float :
        i, lib = 1, [self._message[0]]
        while i < len(self._message):
            for j in range(i, len(self._message)):
                message_ = self._message[i:j + 1]
                if message_ not in lib:
                    lib.append(message_);
                    break
            i = j + 1
        entropy = len(lib) / len(self._message)
        return entropy
    def plug_in(self) -> float :
        if self._word_length is None:
            self._word_length = 1
        pmf = prob_mass_function(self._message, self._word_length)
        out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / self._word_length
        return out
    def konto(self, window: int = 0) -> float :
        out = {
            'h': 0,
            'r': 0,
            'num': 0,
            'sum': 0,
            'sub_str': []
        }
        if window <= 0:
            points = range(1, len(self._message) // 2 + 1)
        else:
            window = min(window, len(self._message) // 2)
            points = range(window, len(self._message) - window + 1)
        for i in points:
            if window <= 0:
                length, msg_ = match_length(self._message, i, i)
                out['sum'] += np.log2(i + 1) / length
            else:
                length, msg_ = match_length(self._message, i, window)
                out['sum'] += np.log2(window + 1) / length
            out['sub_str'].append(msg_)
            out['num'] += 1
        try:
            out['h'] = out['sum'] / out['num']
        except ZeroDivisionError:
            out['h'] = 0
        out['r'] = 1 - out['h'] / np.log2(len(self._message))
        return out['h']

class ContinuousEntropy:
    def __init__(self, ret: pd.DataFrame, period: int):
        self.ret = ret
        self.period = period

    def corr(self):
        return self.ret.corr()

    def optimize_bins(self, correlation: bool = False) -> int:
        len_ret = len(self.ret)
        if correlation == False:
            z = (8 + 324 * len_ret + 12 * (36 * len_ret + 729 * len_ret ** 2) ** 0.5) ** (1 / 3)
            b = round(z / 6 + 2 / (3 * z) + 1 / 3)
        else:
            b = round(2 ** (-0.5) * (1 + (1 + 24 * len_ret / (1 - self.corr() ** 2)) ** 0.5) ** 0.5)
        return int(b)

    def continuous_entropy(self, correlation: bool = False) -> pd.DataFrame:
        bin = self.optimize_bins(correlation = correlation)
        etp = []
        for i in range(self.period, len(self.ret)):
            hX = ss.entropy(np.histogram(self.ret[i - self.period:i], bin)[0])
            etp.append(hX)
        etp = pd.DataFrame(etp, index = self.ret.index[self.period:])
        etp.columns = ['Continuous entropy']
        return etp


def shannon_entropy(message : str) -> float :
    exr = {}
    entropy = 0
    for i in message :
        try : exr[i] += 1
        except KeyError : exr[i] = 1
    textlen = len(message)
    for value in exr.values() :
        freq = 1 * value / textlen
        entropy += freq * math.log(freq) / math.log(2)
    entropy *= -1
    return entropy
def lempel_ziv_entropy(message : str) -> float :
    i, lib = 1, [message[0]]
    while i < len(message) :
        for j in range(i, len(message)) :
            message_ = message[i:j+1]
            if message_ not in lib :
                lib.append(message_); break
        i = j + 1
    entropy = len(lib) / len(message)
    return entropy
def plug_in_entropy(message : str, word_length : int = None) -> float :
    if word_length is None :
        word_length = 1
    pmf = prob_mass_function(message, word_length)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length
    return out
def prob_mass_function(message : str, word_length : int) -> dict :
    lib = {}
    if not isinstance(message, str) :
        message = ''.join(map(str, message))
    for i in range(word_length, len(message)) :
        message_ = message[i - word_length : i]
        if message_ not in lib : lib[message_] = [i - word_length]
        else : lib[message_] = lib[message_] + [i - word_length]
    pmf = float(len(message) - word_length)
    pmf = {i : len(lib[i]) / pmf for i in lib}
    return pmf
def match_length(message : str, start_index : int, window : int) -> Union[int, str]:
    sub_str = np.empty(shape=0)
    for length in range(window):
        msg1 = message[start_index: start_index + length + 1]
        for j in range(start_index - window, start_index):
            msg0 = message[j: j + length + 1]
            if len(msg1) != len(msg0): continue
            if msg1 == msg0:
                sub_str = msg1
                break
    return len(sub_str) + 1, sub_str
def konto_entropy(message: str, window: int = 0) -> float:
    out = {
        'h': 0,
        'r': 0,
        'num': 0,
        'sum': 0,
        'sub_str': []
    }
    if window <= 0:
        points = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        points = range(window, len(message) - window + 1)
    for i in points:
        if window <= 0:
            length, msg_ = match_length(message, i, i)
            out['sum'] += np.log2(i + 1) / length
        else:
            length, msg_ = match_length(message, i, window)
            out['sum'] += np.log2(window + 1) / length
        out['sub_str'].append(msg_)
        out['num'] += 1
    try:
        out['h'] = out['sum'] / out['num']
    except ZeroDivisionError:
        out['h'] = 0
    out['r'] = 1 - out['h'] / np.log2(len(message))
    return out['h']