class Rect:
    def __init__(self, width, height):
        self.width = width
        self._height = height

    def area(self):
        return self.width * self._height


if __name__ == '__main__':
    rect = Rect(20, 30)
    print(rect.area())
