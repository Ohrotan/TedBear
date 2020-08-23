"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from init import app
from jinja2 import Template
import ast
from flask import Flask, jsonify, request, session, redirect
from sqlalchemy import create_engine, text



@app.route('/')
@app.route('/home')
def home():
    originlist=app.database.execute('SELECT topics,title,speaker,image,description FROM talks limit 6').fetchall()
    sqllist=[]
    for i in originlist:
        i=list(i)
        i[0]=ast.literal_eval(i[0])
        sqllist.append(i)

    return render_template(
        'home.html',
        name=session['name'],
        sqllist=sqllist
    )
@app.route('/history')
def history():

    originlist = app.database.execute('SELECT a.talks_title,a.last_time,b.topics,b.speaker,b.image,b.duration FROM watching_record as a join talks as b on b.id=a.talks_id '
                                      'where a.user_id=%s',(82)).fetchall()
    sqllist = []
    for i in originlist:
        i=list(i)
        i[2] = ast.literal_eval(i[2])
        sqllist.append(i)
        print(i[1],i[5])
    """Renders the home page."""
    return render_template(
        'history.html',sqllist=sqllist
    )
@app.route('/shadowing')
def shadowing():
    print(app.database.execute('SELECT * FROM user').fetchall())
    """Renders the home page."""
    return render_template(
        'shadowing.html'
    )
@app.route('/logout')
def logout():
    session['id'] = None
    session['name'] = None
    session['email'] = None
    return redirect('/home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        msg = ''
        email = request.form['remail']
        pwd = request.form['rpwd']
        #print(email,pwd)
        account = app.database.execute("SELECT * FROM user WHERE email = %s and pwd = %s",(email,pwd)).fetchall()
        print(account)
        if account:
            session['id']=account[0][0]
            session['email']=account[0][1]
            session['name']=account[0][2]
            return redirect('/home')
        else:
            return render_template('login.html', msg='Incorrect username/password!')

    if request.method == 'GET':
        return render_template('login.html')

@app.route('/register')
def register():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
