import urllib.request

if __name__ == '__main__':
    respose = urllib.request.urlopen('http://www.baidu.com')
    print(respose.read())
