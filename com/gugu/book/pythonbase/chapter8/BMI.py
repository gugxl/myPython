def fun_bmi(name, height, weight):
    print('name:' + name + ', height:' + str(height) + ', weight:' + str(weight))
    print('bmi=' + str(weight / (height * height)))

def fun_bmi_upgrade(*name):
    fun_bmi(name, 1.73, 67)
