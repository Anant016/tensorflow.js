//1.
var tense = tf.tensor([1, 2, 3, 4]);
tense.print();
console.log(tense);
console.log(tense.toString());

//2.
tf.tensor([1, 2, 3, 4.5], [2, 2]).print();
tf.tensor([1, 2, 3, 4, 5.6, 6, 7, 8], [2, 2, 2], "int32").print();

//3.
tf.tensor(4).print();
tf.scalar(4).print();

var values = [1, 2, 3, 4.5];
var values2 = [1, 2, 3, 4, 5, 6, 7, 8];
tf.tensor1d(values).print();
tf.tensor2d(values, [2, 2]).print();
var tensor3 = tf.tensor3d(values2, [2, 2, 2]);
tensor3.print();

//4.
console.log(tense.data()); //GPU
tense.data().then(function (stuff) {
  console.log(stuff);
});

//5.
console.log(tensor3.dataSync());

//6. VARIABLE
const vtense = tf.variable(tense).print();

//Aruthmetic operations
// addEventListener, mul,div,transpose,etc.

//7. MM
console.log(tf.memory().numTensors);
tensor3.dispose();
console.log(tf.memory().numTensors);

tf.tidy(tidycontent);
function tidycontent() {
  // const a=
  // const b=
  // ....
}
tf.tidy(() => {});
