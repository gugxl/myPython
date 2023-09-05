if __name__ == '__main__':
    aa = num = 1111
    print(id(num))
    print(id(aa  ))
    num = 1112
    print(id(num))
    num = False
    if (num or not num):
        print(num)

