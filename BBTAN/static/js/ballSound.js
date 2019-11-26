class BallSound {
  constructor()
  {
    this.coin = new Audio('static/sounds/coin.mp3');
    this.powerUpLaser = new Audio('static/sounds/laser.mp3');
    this.powerUpSplit = new Audio('static/sounds/split.mp3');
    this.addBall = new Audio('static/sounds/addBall.mp3');
    this.collision = new Audio('static/sounds/collision.mp3');
  }

  play(element){
    if (!PLAY_SOUND) return;

    if (element == 'coin') {
      this.coin.currentTime = 0;
      this.coin.play();
    } else if (element == 'powerUpLaser') {
      this.powerUpLaser.currentTime = 0;
      this.powerUpLaser.play();
    } else if (element == 'powerUpSplit') {
      this.powerUpSplit.currentTime = 0;
      this.powerUpSplit.play();
    } else if (element == 'addBall') {
      this.addBall.currentTime = 0;
      this.addBall.play();
    } else if (element == 'collision') {
      this.collision.currentTime = 0;
      this.collision.play();
    }
  }
}
