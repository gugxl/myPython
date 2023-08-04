if __name__ == '__main__':
    list = list(range(10))
    # del list
    print(list)

    for index, item in enumerate(list):
        print(index, item)

    list.extend([10,20])
    print(list)

    if 20 in list:
        list.remove(20)
    print(list)

    print(sum(list))