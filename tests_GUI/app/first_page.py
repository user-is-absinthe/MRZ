# -*- coding: utf-8 -*-
import sys

from app import app
from flask import render_template

# @app.route('/')
@app.route('/first')

def first():
    user = {'username': 'Max'}
    return render_template('first.html', title=sys._getframe().f_code.co_name, user=user)
    #  title=sys._getframe().f_code.co_name - возвращает имя текущей функции def