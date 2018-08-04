import xml.dom.minidom as xmlparser
import pymysql
import warnings
def ConnectDB():
    #READ PARAMETERS
    dom=xmlparser.parse("../data/dbSetup.xml")
    config=dom.documentElement
    host=config.getElementsByTagName("host")[0].childNodes[0].nodeValue
    port=eval(config.getElementsByTagName("port")[0].childNodes[0].nodeValue)
    user = config.getElementsByTagName("user")[0].childNodes[0].nodeValue
    password = config.getElementsByTagName("password")[0].childNodes[0].nodeValue
    dbname = config.getElementsByTagName("dbname")[0].childNodes[0].nodeValue
    #GET THE CURSOR OF THE DATABASE
    conn = pymysql.connect(host=host, port=port, user=user, passwd=password,
                            db=dbname, charset="utf8")
    return conn

#for test purpose
def testConnect(cur):
    sql = "select * from challenge_item limit 10;"
    cur.execute(sql)
    rows = cur.fetchall()

    for dr in rows:
        print(dr)


def main():

    testConnect(ConnectDB().cursor())

if __name__=="__main__":
    main()


