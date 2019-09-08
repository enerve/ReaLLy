/*
 * Created on 16 May 2019
 * 
 * @author: enerve
 */

/**
 * Agent that uses an onnx-initialized NN to respond to connect-4 games
 */
Agent = function() {
};

Agent.prototype.loadModel = async function() {
	// create a session
    this.onnxSession = new onnx.InferenceSession();
    // load the ONNX model file
    await this.onnxSession.loadModel("./model/986337v3.onnx")
};
    
Agent.prototype.bestAction = async function(S) {
	const inferenceInputs = this.validBoundActions(S);
	// execute the model for each action
	var best = -999
	var bestA = -1
	var str = ""
	for (var a = 0; a < 7; a++) {
		if (inferenceInputs[a] != null) {
			const output = await this.onnxSession.run([inferenceInputs[a]])
			// consume the output
			const outputVal = output.values().next().value.data[0];
		    
			str += a + ":" + outputVal.toFixed(4) + ", "
			if (outputVal > best) {
		    	best = outputVal;
		    	bestA = a;
		    }
		}
	}
	console.log(`model output: ${str}.`);
	return bestA
};

function clone(S) {
	var C = [];
	for (i=0; i<6; i++) {
		C.push(S[i].slice(0));
	}
	return C;
}

AGENT_COIN = 1

Agent.prototype.validBoundActions = function(B) {
	var boundActions = []
	
	for (let col = 0; col < 7; col++) {
		if (B[5][col] == 0) { // Check validity of action
			for (let h = 0; h < 6; h++) {
				if (B[h][col] == 0) {
					B[h][col] = AGENT_COIN;
					boundActions.push(this.features(B));
					B[h][col] = 0;
					break;
				}
			}
		} else {
			boundActions.push(null);
		}
	}
	
	return boundActions;
};

Agent.prototype.features = function(B) {
	
	var x = new Float32Array(2 * 6 * 7).fill(0);
	var ix = 0
	for (let coin = 1; coin >= -1; coin-=2) {
		for (let i = 0; i < 6; i++) {
			for (let j = 0; j < 7; j++) {
				if (B[i][j] == 0)
					x[ix] = -1;
				else if (B[i][j] == coin)
					x[ix] = 1;
				ix++
			}
		}
	}
	
	return new onnx.Tensor(x, 'float32', [1, 2, 6, 7]);
}