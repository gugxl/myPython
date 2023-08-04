def print_like(*likes):
    for like in likes:
        print(like)
def fun_return(name):
    return "hello "+name
if __name__ == '__main__':
    print_like('a','b')
    print(fun_return("gugu"))