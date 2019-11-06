class ObsTriangleBotLeft {
  constructor(ctx,row,column,game) {
    this.x = (TILE_WIDTH * column) + TILE_PADDING;
    this.y = (TILE_HEIGHT * row) + TILE_PADDING + TOP_HEIGHT;
    this.ctx = ctx;
    this.ctx.font = 'bold 12px Arial';
    this.level = game.level;
    this.offset1 = [0,0];
    this.offset2 = [0,OBSTACLE_HEIGHT];
    this.offset3 = [OBSTACLE_WIDTH,OBSTACLE_HEIGHT];
    this.textX = 4;//aligning font at center
    this.textY = 36;//aligning font at center
    this.ctx.strokeStyle = 'yellow';
    this.ctx.fillStyle = 'yellow';
    this.ctx.lineWidth = LINE_WIDTH;
    this.game = game;
    this.row = row;
    this.column = column;
  }

  drawTriangleBotLeft(level) {
    this.level = level;
    this.ctx.beginPath();
    this.ctx.moveTo(this.x+this.offset1[0],this.y+this.offset1[1]);
    this.ctx.lineTo(this.x+this.offset2[0],this.y+this.offset2[1]);
    this.ctx.lineTo(this.x+this.offset3[0],this.y+this.offset3[1]);
    this.ctx.fillText(this.level,this.textX+this.x,this.textY+this.y);
    this.ctx.closePath();
    this.ctx.stroke();
  }

  checkCollision(ball) {

    var c = new SAT.Circle(new SAT.Vector(ball.x,ball.y), BALL_RADIUS);
    var p = new SAT.Polygon(new SAT.Vector(this.x, this.y), [
      new SAT.Vector(this.offset1[0],this.offset1[1]),
      new SAT.Vector(this.offset3[0],this.offset3[1]),
      new SAT.Vector(this.offset2[0],this.offset2[1])
    ]);
    var response = new SAT.Response();
    let collision = SAT.testPolygonCircle(p,c,response)
    if(collision){
      console.log("Bot Left");
      console.log(response);
    }
    return collision
  }
}