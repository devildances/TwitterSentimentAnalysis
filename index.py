__author__ = 'devildances'

import os
from flask import Flask, render_template, request
from EngineFiles import predictDashboard

template_dir = os.path.abspath('Dashboard/frontend/')
static_dir = os.path.abspath('Dashboard/static/')
app = Flask('__main__', template_folder=template_dir, static_folder=static_dir)

@app.route(
    '/',
    methods=['POST','GET']
)
def index():
    predict, original = None, None
    if request.method == 'POST':
        original = request.form['tweettextinput']
        predict = request.form['tweettextinput']
        submit = request.form['submit']
        if submit == "lr":
            predict = predictDashboard.LR_DashboardPredictionResult(original, freq, theta, alay_dict)
        elif submit == "nb":
            predict = predictDashboard.NB_DashboardPredictionResult(original, logprior, loglikelihood, alay_dict)
        elif  submit == "nn":
            predict = predictDashboard.NN_DashboardPredictionResult(original, load_mdl, alay_dict, vocab)
    return render_template('index.html', output1=original, output2=predict)

if __name__ == '__main__':
    alay_dict, freq, theta, logprior, loglikelihood, vocab, load_mdl = predictDashboard.DashboardPredictionLoad()
    # print(logprior)
    app.run(host='127.0.0.1', port=5000, debug=True)