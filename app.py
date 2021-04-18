from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import send_file
from flask import send_from_directory
from flask import Markup

import jenkinsapi
from jenkinsapi.jenkins import Jenkins

import json
import os, uuid

# Extras
import extras
from datetime import datetime

# AI
import ai

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
    
    
    
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    errorMsgs = []
    
    if request.method == 'POST':
        dir_name = str(uuid.uuid4())
        print("Going to store file in uploads/" + dir_name)
        os.makedirs('uploads/' + dir_name, 0o755)
        
        #LearnCard
        LearnCard = request.files['learncard']
        LearnFile = 'uploads/' + dir_name + '/LC_000_%s'%LearnCard.filename
        #
        LearnCard.save(LearnFile)
        errorMsgs += extras.validateLearnCardFile(LearnFile)['errors']
 
        #InCard
        InCard = request.files['incard']
        InFile = 'uploads/' + dir_name + '/IC_%03i_%s'%(0, InCard.filename)
        #
        InCard.save(InFile)
        errorMsgs += extras.validateInCardFile(InFile, LearnFile)['errors']
        
		#eMail
        userEmail = request.form['email']
        if not extras.validateEmail(userEmail):
            errorMsgs += ['Error: Email not valid for this version!']

        #NetworkParams
        NetworkParams = 'uploads/' + dir_name + '/NetworkParams.json'
        info = {}
        for x in request.form.keys(): info[x] = request.form[x]
        with open(NetworkParams, 'w') as f:
            json.dump(info, f)      

		
        if len(errorMsgs) != 0: 
            message_title = 'Fail!'
            message_body = Markup(  '<br/>'.join(errorMsgs)  )
            return render_template('index.html', show_fail = True, show_success = False, message_title=message_title, message_body=message_body)
        
        ###################### Success
        # Trigger job
        #J = Jenkins('http://localhost:8080', username='admin', password='password', useCrumb=True)
        
        
        TEST = False
        if not TEST:
            J = Jenkins('http://jenkins:8080', username='sequi', password='5werty', useCrumb=True)
            params = {'IN_DIR': dir_name, 'IN_MAIL': userEmail}
            J.build_job('run-ml', params)
        
        else:
            run_process_test(dir_name)
        
        message_title = 'Success!'
        message_body = Markup('We are processing your cards. We will send you the result by e-mail to  <strong>' + userEmail + '</strong>.')
        
        with open( "emails.txt" , "a+" ) as f:
            f.write('%s\t%s\n'%(datetime.now().strftime("%m/%d/%Y %H:%M:%S"), userEmail)  )

    return render_template('index.html', show_fail = False, show_success = True, message_title=message_title, message_body=message_body)

def run_process_test(_id):
    print(' ========== RUN PROCESS ========== ')
    print('%s'%('uploads/' + _id))
    result = ai.process('uploads/' + _id)
    #result='OC_000_InCard_Islands.xlsx'
    
    return

@app.route('/run/<_id>')
def run_process(_id):
    print(' ========== RUN PROCESS ========== ')
    print('%s'%('uploads/' + _id))
    result = ai.process('uploads/' + _id)
    #result='OC_000_InCard_Islands.xlsx'
    
    return send_file('uploads/' +_id + r'/%s'%result, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
	
	
@app.route('/results', methods=['GET', 'POST'])
def results():
    print('RESULTS')
    
    _id = request.args.get('id', default = '', type = str)
    dir_name = 'uploads/' + _id
    
    incards = extras.InCardsFiles(dir_name)
    outcards = extras.OutCardsFiles(dir_name)
    learncard = extras.LearnCardFile(dir_name)
    
    #for rendering
    from extras import uploadedFilesFilenameSize as S
    _incards = [x[S:] for x in incards]
    _outcards = ['out_' + x[S:] for x in outcards]
    _learncard = learncard[S:]
    
    #_id = dir_name
    #_incards = ['in1', 'in2']
    #_outcards = ['q','e']
    #_learncard = 'sarasa'
    
    computed = [extras.nCardFile(x) for x in outcards]
    print('COMPUTED = ', computed)
    for ICFILE in [x for x in incards if extras.nCardFile(x) not in computed]:
        ai.loadModelAndProcess(dir_name, ICFILE)

    return render_template('results.html', _id_ = _id, incards=_incards, outcards=_outcards, learncard=_learncard, processing=len(_incards)>len(_outcards))

@app.route('/newincard', methods=['GET', 'POST'])
def newincard():
    print('NEWINCARD')
    errorMsgs = []
    from extras import uploadedFilesFilenameSize as S
    
    if request.method == 'POST':
        _id = request.args.to_dict()['id']
        dir_name = 'uploads/' + _id
        InCard = request.files['incard']
    
    nInCards = len( extras.InCardsFiles(dir_name) )
    InFile = dir_name + r'/IC_%03i_%s'%(nInCards, InCard.filename)
    #
    InCard.save(InFile)
    LearnFile = dir_name + r'/%s'%extras.LearnCardFile(dir_name)
    errorMsgs += extras.validateInCardFile(InFile, LearnFile)['errors']
    
    print(errorMsgs)
    if len(errorMsgs) == 0:
        return redirect('results?id=%s'%_id)
    else:
        os.remove(InFile)
        #for rendering
        _incards = [x[S:] for x in extras.InCardsFiles(dir_name)]
        _outcards = ['out_' + x[S:] for x in extras.OutCardsFiles(dir_name)]
        _learncard = extras.LearnCardFile(dir_name)[S:]
        message_body = Markup(  '<br/>'.join(errorMsgs)  )
        
        return render_template('results.html', _id_ = _id, incards=_incards, outcards=_outcards, learncard=_learncard, incardError = True, message_body=message_body)
    
    
@app.route('/getcards', methods=['GET'])
def getcards():
    from extras import uploadedFilesFilenameSize as S
    _folder = 'uploads/' + request.args.get('id', default = '', type = str)
    _item = request.args.get('item', default = '', type = int)
    _type = request.args.get('type', default = '', type = str)
    
    print('FOLDER %s'%_folder)
    print('ITEM %s'%_item)
    print('TYPE %s'%_type)
    
    if _type == 'incard':
        _file = extras.InCardsFiles(_folder)[_item]
        _name = _file[S:]
		
    if _type == 'outcard':
        _file = extras.OutCardsFiles(_folder)[_item]
        _name = 'out_' + _file[S:]
    
    if _type == 'learncard':
        _file = extras.LearnCardFile(_folder)
        _name = _file[S:]

    return send_from_directory(_folder, _file, attachment_filename=_name, as_attachment=True)

@app.route('/uploads/<path:filename>')
def send_img(filename):
    print('te mando este archivo:: %s'%filename)
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
