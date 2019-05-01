import xml.dom.minidom as xmlparser
import pymysql
import tqdm

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
def ConnectTest(cur):
    sql = "select * from challenge_item;"
    cur.execute(sql)
    cur.fetchone()
    rows = cur.fetchall()
    data=[]
    count=10
    for dr in tqdm.tqdm(rows):
        data.append(dr)
        if len(data)<count:
            print(dr)
    print(len(data))
def main():

    ConnectTest(ConnectDB().cursor())

if __name__=="__main__":
    main()


