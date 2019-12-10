# CSCI 2470 Deep Learning Project

- Title: _Mastering BBTAN with Deep Reinforcement Learning_
- Team Members: `tjiansin`, `ilim5`, `xli148`, `cwang147`

## Components

- BBTAN Game: `\BBTAN`
   - Original Source Code: https://github.com/leapfroglets/BBTan
- Web Bridge: `bridge.py`
   - Web bridge receives game state from game in a JSON document in the following format: 
   ```yaml
   {
        level: game.level, 
        balls: game.totalBalls,
        coin: game.coin,
        bot_x: game.bbtanGameBot.x,
        bot_y: game.bbtanGameBot.y,
        tileMap: game.tileMap,
        levelMap: game.levelMap,
        gameTime: game.gameTime,
        gameStatus: game.gameStatus
    }
   ```
   - Web bridge cleans and flattens game state and sends it to Core
   ```yaml
   [normalized_balls normalized_bot_x normalized_tileMap levelMap]
   index 0: balls
   index 1-352: bot_x
   index 353-1172: tileMap
   index 1173-1236: levelMap
   ```
- Model Selector: `selector.py`
- A2C Runner: `A2C_runner.py`
	- Contains core logic for training
- A2C Model: `A2C_model.py`

## Instructions

Create a new virtualenv, and install requirements:

```sh
pip install -r requirements.txt
```

Start the Flask server:

```sh
python bridge.py
```

Open the game in the browser at <http://127.0.0.1:8000>, which will begin model training.

## Game Engine Optimizations

### Disable Renders, Animations, Sounds

The game code has been instrumented to check various boolean flags before rendering updates to the HTML5 Canvas, or
before playing a sound. This can help aid training.

The flags can be individually controlled in `constants.js`.
