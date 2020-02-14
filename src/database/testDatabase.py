from src.database.MongoDatabase import Database


mongo = Database()
db = mongo.connectToDatabase()
# mongo.checkDatabaseConection()
mongo.readUserInformation()
user = mongo.getUser(fin_code='5KAHJCM')
for us in user:
    print(us)
