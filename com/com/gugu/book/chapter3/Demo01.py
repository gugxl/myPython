if __name__ == '__main__':
    num = 0
    flag = True
    while flag:
        num = num + 1
        if num % 3 == 2 and num % 5 == 3 and num % 7 == 2:
            print("数:", num)
            flag = False
