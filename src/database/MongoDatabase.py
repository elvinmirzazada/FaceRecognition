from pymongo import MongoClient
from pprint import pprint

class Database:

    database = None

    def __init__(self, url = 'localhost', port = 27017):
        self.url = url
        self.port = port
        self.connectToDatabase()


    def connectToDatabase(self):
        self.database = MongoClient(host=self.url, port = self.port).fog

    def checkDatabaseConection(self):
        serverStatusResult=self.database.command("serverStatus")
        pprint(serverStatusResult)


    def readUserInformation(self):
        return (self.database.person.find({}))


    def getUser(self, fin_code):
        userInfo = self.database.person.find({'fin_code': fin_code})
        return userInfo
