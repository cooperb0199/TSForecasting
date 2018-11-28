# -*- coding: utf-8 -*-

import pyodbc
from rawTwitterObject import RawTwitterObject
from twitterAPIClient import TwitterAPIClient

class DatabaseInterface:
    server = ''
    database = ''
    username = ''
    password = ''
    #driver= '/usr/local/Cellar/msodbcsql17/17.0.1.1/lib/libmsodbcsql.17.dylib'
    driver = '{ODBC Driver 13 for SQL Server}'
    
    def __init__(self):
        self.cnxn = pyodbc.connect('DRIVER='+self.driver+';PORT=1433;SERVER='+self.server+';PORT=1443;DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password)
        self.cursor = self.cnxn.cursor()
    
    def AddNewMessage(self, rawTwitterObject):
        self.cursor.execute("""
                       INSERT INTO RawTwitterData
                       VALUES
                       (%s,%s,%s,%s,%s)
                       """,
                       (rawTwitterObject.TwitterId,
                        rawTwitterObject.Nickname,
                        rawTwitterObject.MessageId,
                        rawTwitterObject.Message,
                        rawTwitterObject.TimeStamp))
        
    def Test(self):
        self.cursor.execute("""
                            SELECT * FROM RawTwitterData
                            """)
        
x = TwitterAPIClient()
x.GetTweets("helloiconworld", "icon", 30)
y = DatabaseInterface()
for twitterObject in x.id2text:
    y.AddNewMessage(twitterObject)
y.Test()   
row = y.cursor.fetchone()
while row:
    print (str(row[0]) + " " + str(row[1]))  
    row = y.cursor.fetchone()
#        cursor.execute("SELECT TOP 20 pc.Name as CategoryName, p.name as ProductName FROM [SalesLT].[ProductCategory] pc JOIN [SalesLT].[Product] p ON pc.productcategoryid = p.productcategoryid")
#    row = cursor.fetchone()
#    while row:
#        print (str(row[0]) + " " + str(row[1]))
#    row = cursor.fetchone()