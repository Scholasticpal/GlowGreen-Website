import os
import secrets
from PIL import Image
from logging import debug
from flask import request,render_template,url_for,redirect,request,abort,Response
from sollutionChallenge import app
from sollutionChallenge.utils import access_camera
import numpy as np
import pandas as pd

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html',title="SollutionChallenge",home="active")

@app.route('/contactus')
def contact():
    return render_template("contactus.html",title="Team")

@app.route('/inference')    
def inference():
    return Response(access_camera.gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
       