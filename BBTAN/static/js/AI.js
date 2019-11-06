function requestAIAction(game){
    console.log("requestAIAction");
    fetch("/get_AI_Action", { 
        method: "POST",
        body: JSON.stringify(
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
            }),
        headers: { "Content-Type": "application/json; charset=utf-8" }
    })
    .then(res => res.json())
    .then(response => {
        //simulate click(shoot ball)
        let rect = game.canvas.getBoundingClientRect();
        var simulateClick = function (elem) {
            // Create our event (with options)
            var evt = new MouseEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: response["mouse_x"] + rect.left,
                clientY: response["mouse_y"] + rect.top,
            });
            // If cancelled, don't dispatch our event
            var canceled = !elem.dispatchEvent(evt);
        };

        if(game.gameStatus == "inGame"){
            simulateClick(game.canvas);
        } else if (game.gameStatus == "gameOver"){
            //restart game
            game.reset();
            game.updateTileMap();
            game.gameSound.play('button');
        }
            
    })
    .catch(err => {
        alert("AI Error")
        console.log(err);
    });

    
}