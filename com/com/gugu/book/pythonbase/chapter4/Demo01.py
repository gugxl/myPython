if __name__ == '__main__':
    arr = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    print(arr[-1])
    for i in range(len(arr)):
        print(arr[i])

    print(arr[1:5])
    print(arr[1:5:2])

    print(arr[1:5] + arr[1:5:2])

    print(arr[1:5] * 5)

    print(3 in arr[1:5])

    arr = [7, 2, 6, 3, 4, 5, 1]
    print(sorted(arr))
