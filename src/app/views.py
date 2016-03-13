from flask import render_template
from app import app
import json
from flask import request
import random
from classification.zb_math_classifier import zbMathClassifier

conf = json.load(open("conf/application.json", "r"))

clf = zbMathClassifier()
clf.initialize(conf['input-dir'])

@app.route('/')
def index():
    return render_template('index.html', title='Paper klassifizieren')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        title = request.form['title']
        abstract = request.form['abstract']
    except:
        abort(400)

    classifications = clf.classify(title, abstract)
    print classifications

    return json.dumps({"success": True, "classes": classifications})

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response