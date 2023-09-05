import pymysql


def showVersion(cursor):
    cursor.execute('select version()')
    data = cursor.fetchone()
    print(data)

def createTable(cursor):
    cursor.execute('DROP TABLE IF EXISTS books')

    sql = """CREATE TABLE books (
    id  int NOT NULL AUTO_INCREMENT,
    name varchar(50) NOT NULL ,
    category varchar(50) NOT NULL ,
    price decimal(10, 2) DEFAULT NULL,
    publish_time date DEFAULT NULL,
    primary key (id)
    ) AUTO_INCREMENT = 1 """
    cursor.execute(sql)


def insertData(db):
    sql = """INSERT INTO  books (name,category, price, publish_time) 
    values (%s,%s,%s,%s)"""
    data = [('java', '计算机', '30.92', '2015-05-27'),
            ('python', '计算机', '40.92', '2015-05-27'),
            ]
    cursor.executemany(sql, data)
    db.commit()


if __name__ == '__main__':
    db = pymysql.connect(host='127.0.0.1', user='root', password='root', port=3307, database='mybatis_plus')
    cursor = db.cursor()
    # showVersion(cursor)
    # createTable(cursor)
    insertData(db)

    db.close()



