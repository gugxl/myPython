class Fruit:
    def __init__(self, color = '绿色'):
        print('水果 ')
    def harvest(self, color):
        print('color:' + Fruit.color)
        print('长大')
        print('color:' + color)

class Apple(Fruit):
    def __init__(self, color = '青色'):
        super().__init__(color)
        self.color = color

    def harvest(self, color):
        print('color:' + self.color)
        print('长大')
        print('color:' + color)


if __name__ == '__main__':
    apple = Apple()
    apple.harvest('红色')
