import { Value, ensureValue } from 'microgradjs/engine';
import { range } from 'microgradjs/utils';
import { MLP } from 'microgradjs/nn';

function sample(population, k) {
  /*
      Chooses k unique random elements from a population sequence or set.

  */
  if (!Array.isArray(population))
    throw new TypeError('Population must be an array.');
  var n = population.length;
  if (k < 0 || k > n)
    throw new RangeError('Sample larger than population or is negative');

  var result = new Array(k);
  var setsize = 21; // size of a small set minus size of an empty list

  if (k > 5) setsize += Math.pow(4, Math.ceil(Math.log(k * 3) / Math.log(4)));

  if (n <= setsize) {
    // An n-length list is smaller than a k-length set
    var pool = population.slice();
    for (var i = 0; i < k; i++) {
      // invariant:  non-selected at [0,n-i)
      var j = (Math.random() * (n - i)) | 0;
      result[i] = pool[j];
      pool[j] = pool[n - i - 1]; // move non-selected item into vacancy
    }
  } else {
    var selected = new Set();
    for (var i = 0; i < k; i++) {
      var j = (Math.random() * n) | 0;
      while (selected.has(j)) {
        j = (Math.random() * n) | 0;
      }
      selected.add(j);
      result[i] = population[j];
    }
  }

  return result;
}

function loss(X, y, model, lr, alpha, batch_size) {
  var X_b, y_b;
  if (batch_size != null) {
    const indexes = sample(Array.from(Array(X.length).keys()), batch_size);
    X_b = indexes.map(e => X[e]);
    y_b = indexes.map(e => y[e]);
  } else {
    X_b = X;
    y_b = y;
  }
  const inputs = X_b.map(function (row) {
    return row.map(x => ensureValue(x));
  });

  let scores = inputs.map(row => model.call(row));
  // svm max margin loss
  let losses = range(0, scores.length).map(function (i) {
    return scores[i][0]
      .mul(-y_b[i])
      .add(+1.0)
      .relu();
  });
  let data_loss = losses
    .reduce((sum, current) => sum.add(current), new Value(0))
    .mul(1 / losses.length);
  let reg_loss = model
    .parameters()
    .map(e => e.mul(e))
    .reduce((sum, cur) => sum.add(cur), new Value(0))
    .mul(alpha);
  let total_loss = data_loss.add(reg_loss);

  const accuracies = range(0, scores.length).map(function (i) {
    return scores[i][0].data > 0.0 === y_b[i] > 0.0 ? 1.0 : 0.0;
  });
  const accuracy =
    accuracies.reduce((sum, current) => sum + current, 0.0) / accuracies.length;

  return [data_loss, reg_loss, total_loss, accuracy];
}

function do_epoch(
  k: number,
  model: any,
  X: any,
  y: any,
  lr,
  alpha,
  batch_size,
) {
  // # forward
  let [data_loss, reg_loss, total_loss, acc] = loss(
    X,
    y,
    model,
    lr,
    alpha,
    batch_size,
  );

  // # backward
  model.zero_grad();
  total_loss.backward();

  // # update (sgd)
  let learning_rate = 1.0 - (0.9 * k) / 100;
  // let learning_rate = 0.001;
  for (let p of model.parameters()) {
    // p.data -= lr * p.grad;
    p.data -= learning_rate * p.grad;
  }

  if (k % 1 === 0) {
    console.log(`step ${k} loss ${total_loss.data}, accuracy ${acc * 100}%`);
  }
  return [total_loss.data, acc * 100];
}

function train(
  nbEpochs,
  model,
  X,
  y,
  lr = 0.0001,
  alpha = 0.0001,
  batch_size = null,
) {
  console.log('Start training');
  console.log(
    `epochs=${nbEpochs}, lr=${lr}, alpha=${alpha}, batch_size=${batch_size}`,
  );
  for (let k of range(0, nbEpochs)) {
    do_epoch(k, model, X, y, lr, alpha, batch_size);
  }
}
