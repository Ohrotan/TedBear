from os import listdir

"""
Routes and views for the flask application.
"""
import os
from flask import render_template
from init import app, db_session
from models import Talks, WatchingRecord, Sentence, ShadowingRecord
import ast
from flask import Flask, jsonify, request, session, redirect
import hydra
import uuid
import asyncio

from zerospeech import convert
from zerospeech import editconfig
from evalspeech import evaluate
from zerospeech import preprocess
from zerospeech import train
from googletrans import Translator
from voiceClassify import user_in_speaker

def get_converted_audio(user_id, user_audio_path, org_audio_path, start_transcript, end_transcript):  # 아래 함수들을 한번에 실행
    editconfig.speaker_json(user_audio_path, org_audio_path)
    editconfig.train_json(user_audio_path)
    editconfig.test_json(org_audio_path)
    editconfig.synthesis_json(user_id, org_audio_path, start_transcript, end_transcript)
    convert.modelname(user_id)
    convert.convert()
    hydra._internal.hydra.GlobalHydra().clear()


@app.route('/')
@app.route('/home')
def home():
    session['id'] = 1
    session['name'] = 'dev-test'
    has_transcript_id = "SELECT distinct(talks_id) FROM sentence ORDER BY talks_id"

    sql_query = "SELECT * FROM talks WHERE id in " + has_transcript_id
    valid_talks_id = app.database.execute(has_transcript_id).fetchall()

    valid_talks_id = [a['talks_id'] for a in valid_talks_id]

    originlist = Talks.query.filter(Talks.id.in_(valid_talks_id)).all()

    video_list = []
    for talks in originlist:
        talks.topic = ast.literal_eval(talks.topics)[0]
        video_list.append(talks)

    try:
        if session['name']:
            return render_template(
                'home.html',
                name=session['name'],
                video_list=video_list
            )
    except:
        return render_template(
            'home.html',
            video_list=video_list
        )


@app.route('/history')
def history():
    user_id = 0
    try:
        user_id = session['id']
    except:
        return redirect('/home')

    originlist = db_session.query(WatchingRecord, Talks).filter(
        WatchingRecord.user_id == user_id, WatchingRecord.talks_id == Talks.id).all()
    # watching_list = WatchingRecord.query.join(Talks, WatchingRecord.talks_id == Talks.id).filter(
    #    WatchingRecord.user_id == user_id).all()

    # originlist = app.database.execute(
    #    'SELECT a.talks_title,a.last_time,b.topics,b.speaker,b.image,b.id FROM watching_record as a join talks as b on b.id=a.talks_id '
    #    'where a.user_id=%s', (user_id)).fetchall()
    percent = []
    watching_list = []
    for info in originlist:
        talks_info = info[1]
        talks_info.topic = ast.literal_eval(talks_info.topics)[0]
        watching_list.append(talks_info)
        all = Sentence.query.filter_by(talks_id=talks_info.id).count()

        if all == 0:
            percent.append(0)
        else:
            done = ShadowingRecord.query.filter_by(talks_id=talks_info.id, user_id=session['id']).count()
            percent.append(done / all)

        '''
        sentences = list(app.database.execute(
            'SELECT count(*) from sentence where talks_id=%s', (i[5])).fetchall())
        shadows = list(app.database.execute(
            'SELECT count(*) from shadowing_record where talks_id=%s', (i[5])).fetchall())
        i.append(sentences[0][0])
        i.append(shadows[0][0])
        sqllist.append(i)
        print(sqllist)
        '''
    return render_template(
        'history.html', watching_list=watching_list, percent=percent
    )


def audio_upload(file, file_name):
    print("audio upload: ",os.getcwd())
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'wb') as audio:
        file.save(audio)
    if os.path.isfile(file_name):
        # os.path.abspath('./audio.wav')
        print("user audio uploaded at", os.path.abspath(file_name))


@app.route('/eval', methods=['POST'])
def eval(result=None):
    file_id = str(session['id']) + '_' + request.form['talks_id'] + '_' + request.form[
        'transcript_index']
    origin_audio = '../zerospeech/english/test/' + request.form['talks_id'] + '_' + request.form[
        'transcript_index'] + '.wav'
    user_audio = './english/train/voice/' + file_id + '.wav'
    audio_upload(request.files['audio_data'], user_audio)
    # 평가한 이미지파일, 컨버트 결과 파일 DB에 넣기
    result = ''
    try:
        result = evaluate.eval(os.path.abspath(origin_audio), os.path.abspath(user_audio), file_id)

    except:
        print("Eval error")

    s_record = ShadowingRecord(user_id=session['id'], talks_id=request.form['talks_id'], \
                               sentence_id=request.form['sentence_id'], user_audio=os.path.abspath(user_audio)
                               )
    db_session.add(s_record)

    # voice_recorder.js 155line에서 결과 이미지 띄우는 코드 만들기
    print(result)
    sp = result['speed']
    speed = sp[-1] + " : " + str(sp[0])[:4] + "sec "
    if float(sp[1]) > 100:
        speed = speed + "Slower"
    else:
        speed = speed + "Faster"

    st = result['strength']
    strength = st[-1] + " : " + str(st[0])[:2] + " point"
    pi = result['pitch']
    pitch = pi[-1] + " : " + str(pi[0])[:2] + " point"
    wd = result['words']
    ted = ''
    user = ''
    # if s = 틀림

    for i in range(len(wd[4])):
        ted = ted + wd[6][i] + "^^"
        if wd[4][i] == 's':
            user = user + "@" + wd[5][i] + "^^"
        else:
            user = user + wd[5][i] + "^^"

    word_point = wd[3] + " : " + str(wd[2])[:2] + " point"
    sentence = ted + "%%" + user
    print(sentence)
    tot = result['tot']
    tot_point = tot[-1] + " : " + str(tot[0])[:2] + " point"
    return speed + "+++" + strength + "+++" + pitch + "+++" + word_point + "+++" + sentence + "+++" + tot_point + "+++" + file_id


@app.route('/shadowing/<talks_id>/<transcript_index>')
def shadowing(talks_id, transcript_index):
    # talks 정보 가져오기
    talks_info = Talks.query.filter(Talks.id == talks_id).first()
    global transcript
    transcript = Sentence.query.filter_by(talks_id=talks_id).all()
    transcript_index = int(transcript_index)
    # 사용자 히스토리 기록
    try:
        if session['id'] is None or len(transcript) == 0:
            return redirect('/')
    except:
        return redirect('/')

    seen = WatchingRecord.query.filter_by(user_id=session['id'], talks_id=talks_id).count()

    if seen == 0:
        w_record = WatchingRecord(user_id=session['id'], talks_id=talks_id, talks_title=talks_info.title)
        db_session.begin()
        db_session.add(w_record)
        db_session.commit()

    else:
        print('')
        # 봤던거면 마지막 센텐스 위치찾기
        #transcript_index = ShadowingRecord.query.filter_by(user_id=session['id'], talks_id=talks_id).count()

    if talks_info.youtube_gap is None:
        talks_info.youtube_gap = 0
    transcript[transcript_index].sentence_kr = Translator().translate(str(transcript[transcript_index].sentence_en),
                                                                      dest='ko').text

    return render_template(
        'shadowing.html',
        talks_info=talks_info,
        transcript=transcript,
        transcript_index=transcript_index
    )


@app.route('/move_sentence', methods=['POST'])
def next_sentence():
    num = request.form['transcript_index']
    if request.form['next'] == "true":
        num = int(num) + 1
    else:
        num = int(num) - 1
    if num == len(transcript) or num < 0:
        return 'fail'

    result = transcript[num].__dict__
    result['audio'] = ''
    result['sentence_kr'] = Translator().translate(str(result['sentence_en']), dest='ko').text

    try:
        del result['_sa_instance_state']
    except:
        print('view.py move_sentence')

    return result


@app.route('/record', methods=['GET', 'POST'])
def record():
    # 사용자 오디오 파일 서버에 저장
    if request.method == 'POST':
        user_audio = './english/train/voice/' + str(session['id']) + '_' + uuid.uuid4().hex + '.wav'
        audio_upload(request.files['audio_data'], user_audio)
        s_record = ShadowingRecord(user_id=session['id'], talks_id=0, \
                                   sentence_id=0, user_audio=os.path.abspath(user_audio)
                                   )
        db_session.add(s_record)
        return 'ok'
    transcript = Sentence.query.filter_by(talks_id=77).limit(5).all()
    return render_template('record.html', transcript=transcript)


@app.route('/convert', methods=['POST'])
def req_convert():
    # 사용자 목소리로 컨버트 시키기 비동기처리하기
    filename = "/static/result/" + str(session['id']) + "_" + request.form['talks_id'] + '_' + \
               request.form['transcript_index'] + ".wav"
    if os.path.isfile(".." + filename):
        return filename
    get_converted_audio(str(session['id']), './english/train/voice/', './english/test/',
                        request.form['talks_id'] + '_' + request.form['transcript_index'],
                        request.form['talks_id'] + '_' + request.form['transcript_index'])
    return filename


@app.route('/train', methods=['POST'])
def req_train():
    file_list = [f for f in listdir('./english/train/voice/') if f.startswith(str(session['id']) + "_")]
    user_in_speaker.user_in_speaker(session['id'],'./english/train/voice/'+file_list[0])
    editconfig.train_json('./english/train/voice/', session['id'])
    preprocess.userid(session['id'])
    preprocess.preprocess_dataset()
    hydra._internal.hydra.GlobalHydra().clear()
    train.train_model()
    hydra._internal.hydra.GlobalHydra().clear()


@app.route('/logout')
def logout():
    session['id'] = None
    session['name'] = None
    session['email'] = None
    return redirect('/home')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['remail']
        pwd = request.form['rpwd']

        account = app.database.execute("SELECT * FROM user WHERE email = %s and pwd = %s", (email, pwd)).fetchall()

        if account:
            session['id'] = account[0][0]
            session['email'] = account[0][1]
            session['name'] = account[0][2]
            return redirect('/home')
        else:
            return render_template('login.html', msg='Incorrect username/password!')

    if request.method == 'GET':
        return render_template('login.html')


@app.route('/register')
def register():
    return render_template(
        'about.html'
    )

