if __name__ == '__main__':
    for i in range(1, 10):
        print(i)
        i = 10

    while True:
        i = i - 1
        print(i)
        if i < 0:
            break

    for i in range(10):
        if i % 2 == 0:
            print(i)
        else:
            pass
