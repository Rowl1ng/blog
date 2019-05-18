/**
 * Created by luoling on 2017/4/9.
 */
var Neuron = {
    createNew: function(c, weights)
    {
        var neuron = {};
        neuron.c = c;
        neuron.weights = weights;
        neuron.feedForward = function(inputs)
        {
            var sum = 0;
            for(var i = 0; i < inputs.length; i++){
                inputs[i] *= weights[i];
                sum += inputs[i];
            }
            sum += c;
            return this.activate(sum);
        }
        neuron.activate = function(sum)
        {
            return 1/(1 + exp(-1*sum)) - 0.5;
        }
        return neuron;
    }
}