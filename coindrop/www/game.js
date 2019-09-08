/*
 * Created on 16 May 2019
 * 
 * @author: enerve
 */


const USER = 0
const AGENT = 1

/**
 * Manage the Connect-4 game and mechanics
 */
Game = function() {
	this.B = new Array(6);
	for (var i = 0; i < this.B.length; i++) {
	  this.B[i] = new Array(7).fill(0);
	}
	this.movesLeft = 6*7
	
	this.coin = 1
	this.currentPlayer = -1 // Undefined
	this.gameOver = false
	
	this.agent = new Agent()
	this.agent.loadModel().then(() => {
    	// generate model input
    	console.log('loaded onnx')
    	this.handOff(AGENT)
    });
};
	
Game.prototype.isValidMove = function(x, y) {
	return (this.B[y][x]==0 && (y==0 || this.B[y-1][x] != 0) && !this.gameOver)
};

Game.prototype.currentCoin = function() {
	return this.coin
};

Game.prototype.isUserTurn = function() {
	return (this.currentPlayer == USER)
};

Game.prototype.move = function(x, y) {
	if (this.B[y][x]!=0) {
		alert('expected a valid move!')
		return
	}
	this.B[y][x] = this.coin
	
	this.updateState(x, y)
};

Game.prototype.updateState = function(x, y) {
	this.movesLeft--
	if (hasWon(this.B, this.coin, x, y)) {
		alert(this.coin + " just won")
		this.gameOver = true
		return
	} else if (this.movesLeft == 0) {
		alert("draw")
		this.gameOver = true
		return
	}
	
	this.coin = -this.coin;
	this.handOff(1 - this.currentPlayer)
};

Game.prototype.handOff = function(nextPlayer) {
	this.currentPlayer = nextPlayer
	if (this.currentPlayer == AGENT) {
		// Agent's turn to play
		this.agent.bestAction(this.B).then((a) => this.agentAction(a))
	}
};

Game.prototype.agentAction = function(a) {
	var x_ = a
	var y_ = 0
	while (this.B[y_][x_] != 0) {
		y_++
	}
	if (!this.isValidMove(x_, y_)) {
		alert("Agent chose invalid move: " + x_)
		return
	}
	placeCoin(x_, y_, this.coin)
	this.move(x_, y_)
};

var game = new Game();

function hasWon(B, coin, x, y) {
    // Checks if 'coin' has won the game by playing at x,y
    
    if (_count(B, coin, x, y, 0, 1) >= 4)
        return true
    if (_count(B, coin, x, y, 1, 0) >= 4)
        return true
    if (_count(B, coin, x, y, 1, 1) >= 4)
        return true
    if (_count(B, coin, x, y, 1, -1) >= 4)
        return true
    
    return false
}

function _count(B, coin, x, y, a, b) {
    // Counts consecutive coins of 'coin' around x,y along direction a,b
    
    var count = 0
    var x_ = x 
    var y_ = y
    for (i = 0; i < 4; i++) {
        if (B[y_][x_] != coin)
            break
        count++
        x_ += a
        if (x_ < 0 || x_ >= 7)
            break 
        y_ += b
        if (y_ < 0 || y_ >= 6)
            break 
    }
    
    if (count == 0)
        return 0
        
    a = -a
    b = -b
    x_ = x+a
    y_ = y+b
    if (x_ < 0 || x_ >= 7 || y_ < 0 || y_ >= 6)
        return count
    const rem = 4-count
    for (i = 0; i < rem; i++) {
        if (B[y_][x_] != coin)
            break
        count++
        x_ += a
        if (x_ < 0 || x_ >= 7)
            break 
        y_ += b
        if (y_ < 0 || y_ >= 6)
            break
    }
        
    return count
}
