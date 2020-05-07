//1.
const model = tf.sequential();

const hidden = tf.layers.dense({
  units: 4,
  inputShape: [2],
  activation: "sigmoid",
});

const output = tf.layers.dense({
  units: 3,
  activation: "sigmoid",
});

model.add(hidden);
model.add(output);

const sgdOptimizer = tf.train.sgd(0.1);

model.compile({
  optimizer: sgdOptimizer,
  loss: tf.losses.meanSquaredError,
});

//send data
const xs = tf.tensor2d([
  [0, 0],
  [0.5, 0.5],
  [1, 1],
]);
const ys = tf.tensor2d([
  [1, 2.2, 0],
  [2, 3, 3],
  [1, 2, 4],
]);
// const ys = tf.tensor2d([[1], [0.5], [0]]);

//train- fit
async function train() {
  for (let i = 0; i < 10; i++) {
    const res = await model.fit(xs, ys, { shuffle: true, epochs: 10 });
    console.log(res.history.loss[0]);
  }
}

train().then(() => {
  console.log("training Completed");
  //predict
  model.predict(xs).print();
});
