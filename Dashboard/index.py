__author__ = 'devildances'

import os
from flask import Flask, render_template, request

template_dir = os.path.abspath('./frontend/')
static_dir = os.path.abspath('./static/')
app = Flask('__main__', template_folder=template_dir, static_folder=static_dir)

@app.route(
    '/',
    methods=['POST','GET']
)
def index():
    predict = None
    if request.method == 'POST':
        predict = request.form['tweettextinput']
    return render_template('index.html', output=predict)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)