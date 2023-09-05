if __name__ == '__main__':
    list = [1, 5, 7, 2, 6, 4, 3, 8, 9]
    list.sort(reverse=True)
    print(list)

    char = ['cat','Tom','Angelga']
    char.sort()
    print(char)
    char.sort(key=str.lower, reverse=True)
    print(char)

    char2 = sorted(char, key=str.lower)
    print(char2)