class Rect:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height


if __name__ == '__main__':
    rect = Rect(20, 30)
    print(rect.area)
