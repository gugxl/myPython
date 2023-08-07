class Geese:
    '''大雁'''
    name = None
    _age = 18
    # 构造方法
    def __init__(self, age, name):
        self._age = age
        self.name = name
        print('大雁')

    def getAge(self):
        return self._age
    def getName(self):
        return self.name


if __name__ == '__main__':
    wild2 = Geese(20, 'gugu')
    print(Geese._age)
    print(wild2.name)
