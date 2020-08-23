"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from init import app
from jinja2 import Template

from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text

@app.route('/')
@app.route('/home')
def home():
    print(app.database.execute('SELECT * FROM user').fetchall())
    """Renders the home page."""
    return render_template(
        'home.html',
        title='Home Page'
    )
@app.route('/history')
def history():
    """Renders the home page."""
    return render_template(
        'history.html'
    )
@app.route('/shadowing')
def shadowing():
    print(app.database.execute('SELECT * FROM user').fetchall())
    """Renders the home page."""
    return render_template(
        'shadowing.html'
    )
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['remail']
        pwd = request.form['rpwd']
        app.database.execute('SELECT * FROM user WHERE email == ?, pwd == ?').fetchall()
        return render_template('')

    if request.method == 'GET':
        return render_template(
            'login.html',
            title='Contact',
            year=datetime.now().year,
            message='Your contact page.'
        )

@app.route('/register')
def register():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
