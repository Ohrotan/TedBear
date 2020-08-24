"""
The flask application package.
"""

from flask import Flask
from sqlalchemy import create_engine, text
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

app = Flask(__name__)
app.config.from_pyfile('config.py')

database = create_engine(app.config['DB_URL'], encoding='utf-8')
app.database = database.connect()
app.secret_key = app.config['KEY']

db_session = scoped_session(sessionmaker(autocommit=True,
                                         autoflush=True,
                                         bind=database))
Base = declarative_base()
Base.query = db_session.query_property()
Base.metadata.create_all(bind=database)
import views
