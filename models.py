from init import Base
from sqlalchemy import Column, Integer, String, Date, Text, Float


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(120))
    pwd = Column(String(120))

    def __init__(self, name=None, email=None):
        self.name = name
        self.email = email

    def __repr__(self):
        return '<User %s, %s, %s>' % (self.id, self.email, self.name)


class Talks(Base):
    __tablename__ = 'talks'
    id = Column(Integer, primary_key=True)
    url = Column(String(100))
    title = Column(String(100))
    speaker = Column(String(100))
    image =Column(String(100))
    published_date = Column(Date)
    image = Column(String(100))
    duration = Column(Integer)
    topics = Column(String(50))
    description = Column(Text)
    youtube_gap = Column(Integer)
    yt_url = Column(String(50))

    def __init__(self, id=None, url=None, title=None, speaker=None, image=None, youtube_gap=None,
                 published_date=None, duration=None, description=None, yt_url=None):
        self.id = id
        self.url = url
        self.title = title
        self.speaker = speaker
        self.youtube_gap = youtube_gap
        self.image = image
        self.published_date = published_date
        self.duration = duration
        self.description = description
        self.yt_url = yt_url

    def __repr__(self):
        return '<Talks %s, %s, %s>' % (self.id, self.title, self.speaker)


class Sentence(Base):
    __tablename__ = 'sentence'
    id = Column(Integer, primary_key=True)
    talks_id = Column(Integer)
    start_time = Column(Float)
    end_time = Column(Float)
    duration = Column(Float)
    audio = Column(String(50))
    sentence_en = Column(String(50))
    sentence_kr = Column(String(50))

    def __init__(self, id=None, talks_id=None, start_time=None, end_time=None, \
                 duration=None, audio=None, sentence_en=None):
        self.id = id
        self.talks_id = talks_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.audio = audio
        self.sentence_en = sentence_en
        self.sentence_kr = sentence_kr

    def __repr__(self):
        return '<Sentence %s, %s, %s>' % (self.id, self.talks_id, self.sentence_en)


class WatchingRecord(Base):
    __tablename__ = 'watching_record'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    talks_id = Column(Integer)
    talks_title = Column(String(50))
    last_time = Column(Date)

    def __init__(self, id=None, user_id=None, talks_id=None, talks_title=None, last_time=None):
        self.id = id
        self.user_id = user_id
        self.talks_id = talks_id
        self.talks_title = talks_title
        self.last_time = last_time

    def __repr__(self):
        return '<WatchingRecord %s, %s, %s>' % (self.id, self.user_id, self.talks_id)


class ShadowingRecord(Base):
    __tablename__ = 'shadowing_record'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    talks_id = Column(Integer)
    sentence_id = Column(Integer)
    user_audio = Column(String(50))
    converted_audio = Column(String(50))
    accent_score = Column(Integer)
    strength_score = Column(Integer)
    speed_score = Column(Integer)
    pronunciation_score = Column(Integer)
    tot_score = Column(Integer)
    accent_img = Column(String(50))
    strength_img = Column(String(50))
    speed_img = Column(String(50))
    pronunciation_img = Column(String(50))

    def __init__(self, id=None, user_id=None, talks_id=None, sentence_id=None, accent_score=None,\
                 strength_score=None, speed_score=None, pronunciation_score=None, tot_score=None, \
                 accent_img=None, strength_img=None, speed_img=None, pronunciation_img=None,\
                 user_audio=None, converted_audio=None):
        self.id = id
        self.user_id = user_id
        self.talks_id = talks_id
        self.sentence_id = sentence_id
        self.user_audio = user_audio
        self.converted_audio = converted_audio
        self.accent_score = accent_score
        self.strength_score = strength_score
        self.speed_score = speed_score
        self.pronunciation_score = pronunciation_score
        self.tot_score = tot_score
        self.accent_img = accent_img
        self.strength_img = strength_img
        self.speed_img = speed_img
        self.pronunciation_img = pronunciation_img

    def __repr__(self):
        return '<ShadowingRecord %s, %s, %s>' % (self.id, self.user_id, self.sentence_id)
