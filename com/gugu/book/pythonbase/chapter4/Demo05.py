if __name__ == '__main__':
    set1 = set([1, 2, 1, 3])
    print(set1)
    set1.add('2')
    print(set1)
    set2 = set(("4444444", "555"))
    print(set2)
    set2 = set(("4444444"))
    print(set2)
    set2 = set({"4444444"})
    set3 = set(["4444444"])
    print(set2)
    print(set3)
    print(set1.pop())

    python = set(['张三', '李四', '王五'])
    java = set(['张三', '李四', '赵柳'])

    print(python - java)
    print(python & java)
    print(python | java)