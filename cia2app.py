from flask import Flask, request, render_template,redirect,url_for
app = Flask(__name__)
import pickle

import pandas as pd
import pymysql as pms
import numpy as np

def model(features):
    import pandas as pd
    data = pd.read_csv("C:\\Users\\Hashim\\OneDrive\\Desktop\\Desktop\\ML\\ML_CIA2\\CIA2_modeldata.csv")
    data = data.drop(columns=["SkinThickness", "Pregnancies","DiabetesPedigreeFunction"])
    
    x = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    
    from sklearn.model_selection import train_test_split
    
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,random_state=14)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression()
    reg.fit(x_train,y_train)
    
    features=np.reshape(np.array(features),(1,-1))
    prediction = reg.predict(features)

# app code
@app.route("/")
def main_1():
    return render_template("CIA2index.html")

@app.route("/login",methods=["POST"])
def validate():
    conn = pms.connect(host="localhost",port=3306,user="root",
                       password="H9895432245H",db="dbms")

    sql="select username,passcode from login;"
    #df=pd.read_sql(sql,conn)
    cur = conn.cursor()
    cur.execute(sql)
    df = cur.fetchall() 
    data=(request.form["usrname"],request.form["psw"])
    print(data)
    if(data in df):
        return render_template("main.html")
    else:
        return redirect(url_for("main_1"))

@app.route("/predict", methods=['post'])
def pred():
    features = [float(i) for i in (request.form.values())]
    print(features)
    pred = model(features)
    if pred==0:
        val="Low"
    else:
        val="High"
    return render_template("final.html",data=val)
    
if __name__=='__main__':
    app.run(host='localhost',port=5000)

pickle.dump(model, open('model.pkl', 'wb'))