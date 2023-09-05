import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    proxy = {'http':'202.169.229.139:53281'}
    resp = requests.get("https://www.baidu.com", proxies=proxy)
    print(resp.status_code)
    print(resp.url)
    print(resp.headers)
    print(resp.cookies)
    print(resp.text)
    print(resp.content)

    soup = BeautifulSoup(resp.text)
    print(soup.prettify())
