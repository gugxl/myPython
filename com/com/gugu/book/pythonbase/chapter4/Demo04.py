if __name__ == '__main__':
    tuple = (1, 2, 3, 4)
    print(tuple[1])
    print(tuple)

    dict0 = {'a': '1', 'b': '2'}

    print(dict0['a'])

    list1 = [1, 2, 3, 4]
    list2 = ['a', 'b', 'c', 'd']

    dict2 = dict(zip(list2, list1))
    print(dict2['b'])

    dict3 = dict(a='1', b='b')
    print(dict3['a'])

    dict4 = dict.fromkeys(['a', 'b', 'c', 'd'])
    print(dict4)

    name_tuple = ('a', 'b', 'c')
    age = [19, 20, 18]
    dict5 = {name_tuple: age}
    print(dict5.get('a', '不存在'))

    for k, v in dict4.items():
        print(k, v)

    dict4['d'] = 5
    if 'c' in dict4:
        del dict4['c']

    print(dict4)
