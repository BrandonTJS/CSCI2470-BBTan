/**
 * FPS counter to benchmark speeding up the game engine.
 */

class FPSCounter {
  constructor(ctx, maxSize) {
    this.ctx = ctx;

    // Initialize circular buffer
    this.counts = Array(maxSize).fill(0);
    this.bufNext = 0;
    this.bufSize = 0;
    this.bufMaxSize = maxSize;

    // Store current sum of elements in buffer
    this.bufSum = 0;

    // Store last measured time
    this.lastTimestamp = null;
  }

  draw() {
    this.ctx.beginPath();
    this.ctx.font = '10px Arial';
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, GAME_HEIGHT - 10, 50, 10);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText(`FPS: ${Math.round(this.getFPS())}`, 0, GAME_HEIGHT);
    this.ctx.closePath();
  }

  // Adds a new counter to the FPS counter.
  frameDone() {
    const currentTimestamp = Date.now();

    // Initialize first timestamp.
    if (this.lastTimestamp === null) {
      this.lastTimestamp = currentTimestamp;
      return;
    }

    // Otherwise, the duration is the time elapsed from the last timestamp.
    const duration = currentTimestamp - this.lastTimestamp;

    // Check if we need to remove old element from circular buffer.
    if (this.bufSize === this.bufMaxSize) {
      this.bufSum -= this.counts[this.bufNext];
    }

    // Add to circular buffer and advance pointer.
    this.counts[this.bufNext] = duration;
    this.bufNext = (this.bufNext + 1) % this.bufMaxSize;
    this.bufSize = Math.min(this.bufSize + 1, this.bufMaxSize);
    this.bufSum += duration;

    // Update last timestamp.
    this.lastTimestamp = currentTimestamp;
  }

  // Gets the current FPS.
  getFPS() {
    // Milliseconds per frame
    const averageFrameDuration = this.bufSum / this.bufSize;

    // Return reciprocal to get frames per second
    return 1 / (averageFrameDuration / 1000);
  }
}
