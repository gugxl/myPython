class TVShow:
    def __init__(self, show):
        self._show = show

    def show(self):
        print(self._show)
    def show2(self, _show):
        self._show = _show

if __name__ == '__main__':
    tv = TVShow('aa')
    tv.show()
    tv.show2('bb')
    tv.show()

