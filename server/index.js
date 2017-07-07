const synaptic = require('synaptic');
const mnist = require('mnist');
const Neuron = synaptic.Neuron,
  Layer = synaptic.Layer,
  Network = synaptic.Network,
  Trainer = synaptic.Trainer,
  Architect = synaptic.Architect;

const inputLayer = new Layer(784);
const hiddenLayer = new Layer(100);
const outputLayer = new Layer(10);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const myNetwork = new Network({
  input: inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
});

const set = mnist.set(700, 20);

const trainingSet = set.training;
const testSet = set.test;

const trainer = new Trainer(myNetwork);
trainer.train(trainingSet, {
  rate: .1,
  iterations: 10,
  error: .01,
  shuffle: true,
  log: 1,
  cost: Trainer.cost.CROSS_ENTROPY
});

console.log(myNetwork.activate(testSet[0].input));
console.log(testSet[0].output);