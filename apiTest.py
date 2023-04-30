from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask_jsonpify import jsonify
app = Flask(__name__)
api = Api(app)
class Test(Resource) :
    def get(self):
        return "123"
api.add_resource(Test,'/test/')

@app.route("/")
def home():
    return "123"
if __name__ == '__main__':
     app.run(debug=True ,port=8080,use_reloader=False)