from flask import Flask
app = Flask(__name__)

@app.route("/")  # 网站根目录(默认为localhost:5000)
def index():
    return "Index!"

@app.route("/hello")
def hello():
    return "Hello World!"

@app.route("/members")
def members():
    return "Members"

@app.route("/members/<string:name>/")  # 该路径支持一个名字作为参数
def getMember(name):
    return name  # 在该网址print出名字

if __name__ == "__main__":
    app.run()
