"""
The flask application package.
"""

from flask import Flask
from sqlalchemy import create_engine, text

app = Flask(__name__)
app.config.from_pyfile('config.py')

database = create_engine(app.config['DB_URL'], encoding='utf-8')
app.database = database
app.secret_key = app.config['KEY']
import views
