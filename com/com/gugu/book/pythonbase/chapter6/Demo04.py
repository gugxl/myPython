import math


def round_area(r1):
    result = lambda r: math.pi * r * r
    return result(r1)

if __name__ == '__main__':
    r = 10
    print(round_area(r))
