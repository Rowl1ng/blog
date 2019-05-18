/**
 * Created by luoling on 2017/4/9.
 */
var Brain = {
    createNew: function(a, b)
    {
        brain = {}
        brain.neuron = new Array();
        brain.weights = new Array();

        //a is inputs length, b is output length
        for (var i = 0; i < a; i++){
            brain.weights[i] = new Array();
            for (var j = 0; j < b; j++){
                brain.weights[i][j] = new Array();
                for (var k = 0; k < b; k++){
                    brain.weights[i][j][k] = random(-1, 1);
                }
            }
        }
        for (var i = 0; i < a; i++) {
            brain.neuron[i] = new Array();
            for (var j = 0; j < b; j++) {
                brain.neuron[i][j] = Neuron.createNew(-0.01, brain.weights[i][j]);
            }
        }

        brain.setWeight = function(newWeight)
        {
            for (var i = 0; i < this.neuron.length; i++) {
                for (var j = 0; j < this.neuron[i].length; j++) {
                    this.neuron[i][j] = Neuron.createNew(-0.01, newWeight[i][j]);
                }
            }
        }
        brain.getWeights = function()
        {
            return this.weights;
        }
        brain.feedForward = function(input)
        {
            var hiddenOut = new Array();
            var action = new Array();
            for (var i = 0; i < this.neuron[0].length; i++)
            {
                hiddenOut[i] = this.neuron[0][i].feedForward(input);
            }
            for (var i = 0; i < this.neuron[1].length; i++)
            {
                action[i] = this.neuron[1][i].feedForward(hiddenOut);
            }
            return action;
        }
        return brain;
    }
}







