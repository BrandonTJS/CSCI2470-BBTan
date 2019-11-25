#!/usr/bin/env python
import os
import hashlib
from flask import Flask, request, render_template, jsonify, redirect, send_from_directory, make_response
import random
import numpy as np
from core import *

app =  Flask(__name__, template_folder='BBTAN')
app._static_folder ='BBTAN/static'

core = Core(state_size=129, action_space=350)
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
	#print("--- GAME STATE ---\n")
	#print(content)
	#print("\n")
    
	#AI code goes here
	#x = random.randint(0,350)
	#y = random.randint(60,60+450)

	if content['gameStatus'] == 'gameOver':
		core.train()
		game_counter += 1
		if game_counter % 5 == 0:
			core.print_total_rewards(num_previous_round=5)
		response = {'status': 'ok'}
		response_code = 200
	elif content['gameStatus'] == 'inGame':
		game_state_flat = []
		game_state_flat.append(content['balls'])
		game_state_flat.append(content['bot_x'])
		game_state_flat.append(content['bot_y'])
		for i in range(len(content['tileMap'])):
			game_state_flat.extend(content['tileMap'][i])
			game_state_flat.extend(content['levelMap'][i])
		#print(game_state_flat)

		action = core.calculate_action(game_state_flat)
		response = {'mouse_x': action.item(), "mouse_y": 300}
		response_code = 200
		print("Predicted Action: (" + str(action) + "," + str(300) + ")\n\n")
	else:
		response = {'error': 'Invalid Game State'}
		response_code = 500

	return jsonify(response), response_code


app.run(debug=True, port=8000)
