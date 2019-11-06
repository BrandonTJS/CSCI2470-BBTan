#!/usr/bin/env python
import os
import hashlib
from flask import Flask, request, render_template, jsonify, redirect, send_from_directory, make_response
import random
import numpy as np

app =  Flask(__name__, template_folder='BBTAN')
app._static_folder ='BBTAN/static'

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/get_AI_Action', methods=['POST'])
def new_transaction():
	content = request.json
	
	#clean levelMap
	tileMap = np.array(content['tileMap'])
	mask = (tileMap == 1) | (tileMap == 2) | (tileMap == 3) | (tileMap == 4) | (tileMap == 5) | (tileMap == 12)
	cleanLevelMap = np.multiply(content['levelMap'], mask)

	#AI code goes here
	response = {'mouse_x': random.randint(0,350), "mouse_y": random.randint(60,60+450)}

	return jsonify(response), 200


app.run(debug=True, port=8000)
