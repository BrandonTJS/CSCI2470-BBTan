# _???_: Mastering BBTAN with Deep Reinforcement Learning

This is a project submission for CSCI 2470 Deep Learning.

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

### Setup

1. Create new virtualenv.
2. Install requirements:

   ```sh
   pip install -r requirements.txt
   ```

3. Start Web Bridge

   ```sh
   python bridge.py
   ```

4. Navigate to 

   ```sh
   http://127.0.0.1:8000
   ```
