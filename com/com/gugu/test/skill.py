import sys

def switch_value():
    x, y = 1, 2
    print(x, y)
    x, y = y, x
    print(x, y)


def str_list_join():
    sentence_list = ["my", "name", "is", "gugu"]
    sentence_string = " ".join(sentence_list)
    print(sentence_string)


def str_split():
    sentence_string = "my name is gugu"
    print(sentence_string.split())


def num_fill_array():
    print([0] * 100)


def dict_merge():
    x = {'a': 1, 'b': 2}
    y = {'b': 3, 'c': 4}
    z = {**x, **y}
    print(z)


def str_reversal():
    name = "hello gugu"
    print(name[::-1])


def return_muti_values():
    a = "gugu"
    b = "is"
    c = "cool"
    return a, b, c


def list_analysis():
    a = [1, 2, 3]
    b = [num * 2 for num in a]
    print(b)

def list_dict():
    m = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    for key, value in m.items():
        print('{0}: {1}'.format(key, value))

def list_array_index():
    m = ['a', 'b', 'c', 'd']
    for index, value in enumerate(m):
        print('{0}: {1}'.format(index, value))

def init_empty_contains():
    a_list = list()
    a_dict = dict()
    a_map = map()
    a_set = set()


def get_list_count_max():
    test = [1, 2, 3, 4, 2, 2, 3, 1, 4, 4, 4]
    print(max(set(test), key=test.count))


def get_object_mem_size():
    x = 1
    print(sys.getsizeof(x))


if __name__ == '__main__':
    'switch_value()'
    'str_list_join()'
    'str_split()'
    'num_fill_array()'
    'dict_merge()'
    'str_reversal()'
    '''sentence = return_muti_values()
    (a, b, c) = sentence
    print(a, b, c)'''
    'list_analysis()'
    'list_dict()'
    'list_array_index()'
    'get_list_count_max()'
    get_object_mem_size()


