if __name__ == '__main__':
    str = 'abcdefg'
    print(str[5:])
    print(str[:5])
    print(str[:])
    print(str[3:5])

    str2 = 'a b c d e f'
    print(str2.split(' ', 3))

    str3 = 'aabbcbbaaccbb'
    print(str3.count('a', 2, 10))
    print(str3.find('a', 2, 10))
    print(str3.index('a', 2, 10))

    str4 = ' abcd '
    print(str4.strip())
    print(str4.strip(' '))
    print(str4.lstrip(' '))