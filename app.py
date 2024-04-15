import json

from flask import Flask, redirect, url_for, request

import os

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Welcome to My Watchlist!'


@app.route('/world')
def hello_world():
    return 'hello world'
    # return '常舒祺别摸鱼了'


@app.route('/guest/<guest>')
def hello_guest(guest):
    return 'hello %s as guest' % guest

@app.route('/absa/ate', methods=['Post'])
def ate():
    if request.method != 'POST':
        return json.dumps({'code': '400', 'msg': 'Require Post Method', 'body': {}}, ensure_ascii=False)
    if 'raw_text' not in request.json:
        return json.dumps({'code': '400', 'msg': 'Wrong Parameters', 'body': {}}, ensure_ascii=False)

    raw_text = request.json.get('raw_text', None)
    text = '"' + raw_text + '"'
    out = os.popen(f"python run_model.py -mode cli -task ate \
        -model_checkpoint ./model/ate_tk-instruct-base-def-pos-neg-neut-combined \
        -test_input {text}")
    ate_output = {}
    for i in out.readlines():
        line = i.replace("\n", "")
        arr = line.split(':  ')
        if len(arr) == 2:
            ate_output[arr[0]] = arr[1]
    return json.dumps({'code': '200', 'msg': 'success', 'body': ate_output}, ensure_ascii=False)


@app.route('/absa/atsc', methods=['Post'])
def atsc():
    if request.method != 'POST':
        return json.dumps({'code': '400', 'msg': 'Require Post Method', 'body': {}}, ensure_ascii=False)
    if 'raw_text' not in request.json or 'aspect_term' not in request.json:
        return json.dumps({'code': '400', 'msg': 'Wrong Parameters', 'body': {}}, ensure_ascii=False)

    raw_text = request.json.get('raw_text', None)
    aspect_term = request.json.get('aspect_term', None)
    text = '"' + raw_text + '|' + aspect_term + '"'
    out = os.popen(f"python run_model.py -mode cli -task atsc \
    -model_checkpoint ./model/atsc_tk-instruct-base-def-pos-neg-neut-combined \
    -test_input {text}")

    atsc_output = {}
    for i in out.readlines():
        line = i.replace("\n", "")
        arr = line.split(':  ')
        if len(arr) == 2:
            atsc_output[arr[0]] = arr[1]
    return json.dumps({'code': '200', 'msg': 'success', 'body': atsc_output}, ensure_ascii=False)


@app.route('/absa/aspe', methods=['Post'])
def aspe():
    if request.method != 'POST':
        return json.dumps({'code': '400', 'msg': 'Require Post Method', 'body': {}}, ensure_ascii=False)
    if 'raw_text' not in request.json:
        return json.dumps({'code': '400', 'msg': 'Wrong Parameters', 'body': {}}, ensure_ascii=False)

    raw_text = request.json.get('raw_text', None)
    text = '"' + raw_text + '"'
    out = os.popen(f"python run_model.py -mode cli -task aspe \
    -model_checkpoint ./model/joint_tk-instruct-base-def-pos-neg-neut-combined \
    -test_input {text}")

    aspe_output = {}
    for i in out.readlines():
        line = i.replace("\n", "")
        arr = line.split(':  ')
        if len(arr) == 2:
            aspe_output[arr[0]] = arr[1]
    # todo: 将Model output中的aspect和情感用json分别返回
    if 'Model output' in aspe_output:
        out = aspe_output['Model output']
        arr = out.split(', ')
        values = {}
        for entry in arr:
            kv = entry.split(':')
            if len(kv) == 2:
                values[kv[0]] = kv[1]
        aspe_output['Model output'] = values
    return json.dumps({'code': '200', 'msg': 'success', 'body': aspe_output}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# text4test = '"The cab ride was amazing but the service was pricey"'
# a = os.popen(f"python run_model.py -mode cli -task aspe \
# -model_checkpoint ./model/joint_tk-instruct-base-def-pos-neg-neut-combined \
# -test_input {text4test}")
# # print(f"a的值为{a}")
# aspe_output = {}
# for i in a.readlines():
#     line = i.replace("\n", "")
#     arr = line.split(':  ')
#     if len(arr) == 2:
#         aspe_output[arr[0]] = arr[1]
# # todo: 将Model output中的aspect和情感用json分别返回
# if 'Model output' in aspe_output:
#     out = aspe_output['Model output']
#     arr = out.split(', ')
#     values = {}
#     # print(arr)
#     for entry in arr:
#         kv = entry.split(':')
#         # print(kv)
#         if len(kv) == 2:
#             values[kv[0]] = kv[1]
#     aspe_output['Model output'] = values
# print(aspe_output)

# python run_model.py -mode cli -task ate \
# -model_checkpoint ./model/ate_tk-instruct-base-def-pos-neg-neut-combined \
# -test_input "The cab ride was amazing but the service was pricey"

# python run_model.py -mode cli -task atsc \
# -model_checkpoint ./model/atsc_tk-instruct-base-def-pos-neg-neut-combined \
# -test_input "The ambience was amazing but the waiter was rude|ambience"
