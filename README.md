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
   [balls, bot_x, bot_y, tileMap, levelMap]
   index 0: balls
   index 1: bot_x
   index 2: bot_y
   index 3-65: tileMap
   index 66-128: levelMap
   ```
- Core Logic: `core.py`
    - Core logic for training and calling model 
- A2C Model: `reinforce_with_baseline.py`

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
