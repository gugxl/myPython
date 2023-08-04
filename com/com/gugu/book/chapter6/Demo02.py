def demo(obj):
    obj += obj


def bmi(weight, height, name='佚名'):
    bmi = weight / (height * height)
    print(name, bmi)

def obj_var(obj=[]):
    print(obj)
    obj.append(1)

def obj_val(obj=None):
    if obj == None:
        obj = []
    print(obj)
    obj.append(1)

if __name__ == '__main__':
    obj = 'gu'
    demo(obj)
    print(obj)
    list = ['a', 'b']
    demo(list)
    print(list)

    bmi(68, 1.73)
    print(bmi.__defaults__)

    obj_var()
    obj_var()

    obj_val()
    obj_val()
