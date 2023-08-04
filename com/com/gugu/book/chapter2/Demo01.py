if __name__ == '__main__':
    height = float(input("输入身高(m)："))
    weight = float(input("输入体重（kg）："))

    # 计算bmi
    bmi = weight/(height*height)

    if bmi < 18.5:
        print("轻" + str(bmi))
    elif bmi > 29.9:
        print("重" + str(bmi))
    else:
        print("正常" + str(bmi))


