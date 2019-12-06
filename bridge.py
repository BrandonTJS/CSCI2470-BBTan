#!/usr/bin/env python
import os
import hashlib
from flask import Flask, request, render_template, jsonify, redirect, send_from_directory, make_response
import random
import numpy as np
from selector import ModelType, ModelSelector

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app =  Flask(__name__, template_folder='BBTAN')
app._static_folder ='BBTAN/static'

model_selector = ModelSelector(ModelType.A2C, 129, 25)
model_selector.load_model()

game_counter = 0

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/get_AI_Action', methods=['POST'])
def new_transaction():
	global core 
	global game_counter 
	content = request.json
	
	#clean levelMap
	tileMap = np.array(content['tileMap'])
	mask = (tileMap == 1) | (tileMap == 2) | (tileMap == 3) | (tileMap == 4) | (tileMap == 5) | (tileMap == 12)
	cleanLevelMap = np.multiply(content['levelMap'], mask)

	#Flatten game state
	game_state_flat = []
	game_state_flat.append(content['balls'])
	game_state_flat.append(content['bot_x'])
	game_state_flat.append(content['bot_y'])
	for i in range(len(tileMap)):
		game_state_flat.extend(tileMap[i])
	for i in range(len(cleanLevelMap)):
		game_state_flat.extend(cleanLevelMap[i])

	#Event Handlers
	if content['gameStatus'] == 'gameOver':
		model_selector.game_over_handler(game_state_flat)
		game_counter += 1
		if game_counter % 500 == 0:
			model_selector.save_model()
		print("Games Played: " + str(game_counter))
		response = {'status': 'ok'}
		response_code = 200
	elif content['gameStatus'] == 'inGame':
		action = model_selector.game_action_handler(game_state_flat)
		response = {'mouse_x': int(action.item())*14, "mouse_y": 470}
		response_code = 200
		print("Predicted Action: (" + str(int(action.item())*14) + "," + str(470) + ")\n")
	else:
		response = {'error': 'Invalid Game State'}
		response_code = 500

	return jsonify(response), response_code


app.run(debug=False, port=8000)
