import numpy as np

def encode_tick_rule_array(tick_rule_array: list) -> str:
    message = ''
    for element in tick_rule_array:
        if element == 1:
            message += 'a'
        elif element == -1:
            message += 'b'
        elif element == 0:
            message += 'c'
        else:
            raise ValueError('Unknown value for tick rule: {}'.format(element))
    return message

def _get_ascii_table() -> list:
    table = []
    for i in range(256):
        table.append(chr(i))
    return table

def quantile_mapping(array: list, num_letters: int = 26) -> dict:
    encoding_dict = {}
    ascii_table = _get_ascii_table()
    alphabet = ascii_table[:num_letters]
    for quant, letter in zip(np.linspace(0.01, 1, len(alphabet)), alphabet):
        encoding_dict[np.quantile(array, quant)] = letter
    return encoding_dict

def sigma_mapping(array: list, step: float = 0.01) -> dict:
    i = 0
    ascii_table = _get_ascii_table()
    encoding_dict = {}
    encoding_steps = np.arange(min(array), max(array), step)
    for element in encoding_steps:
        try:
            encoding_dict[element] = ascii_table[i]
        except IndexError:
            raise ValueError(
                'Length of dictionary ceil((max(arr) - min(arr)) / step = {} is more than ASCII table lenght)'.format(
                    len(encoding_steps)))
        i += 1
    return encoding_dict

def _find_nearest(array: list, value: float) -> float:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def _get_letter_from_encoding(value: float, encoding_dict: dict) -> str:
    return encoding_dict[_find_nearest(list(encoding_dict.keys()), value)]

def encode_array(array: list, encoding_dict: dict) -> str:
    message = ''
    for element in array:
        message += _get_letter_from_encoding(element, encoding_dict)
    return message