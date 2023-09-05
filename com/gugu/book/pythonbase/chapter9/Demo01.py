def division():
    apple = int(input("输入苹果数量:"))
    child = int(input("输入人数:"))

    # if (child > apple):
    #     raise ValueError("苹果不够分")

    assert child <= apple, "苹果不够分"

    res = apple // child
    print("一人几个：" + str(res) + "剩余：" + str(apple % child))


if __name__ == '__main__':
    try:
        division()
    except ZeroDivisionError:
        print("不能0个小朋友！！！")
    except ValueError as e:
        print("输入错误：", e)
    except Exception as e:
        print("其他錯誤：", e)
    else:
        print("正常")
    finally:
        print("结束")
