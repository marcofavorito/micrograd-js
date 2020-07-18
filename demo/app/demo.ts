import { Value, ensureValue } from 'microgradjs/engine';
import { range } from 'microgradjs/utils';
import { MLP } from 'microgradjs/nn';

function loss(X, y, model, batch_size = null) {
  // TODO use batch_size
  const X_b = X;
  const y_b = y;
  const inputs = X_b.map(function (row) {
    return row.map(x => ensureValue(x));
  });
  const labels = y_b.map(function (label) {
    return ensureValue(label);
  });

  let scores = inputs.map(row => model.call(row));
  // svm max margin loss
  let losses = range(0, scores.length).map(function (i) {
    return scores[i][0]
      .mul(-y_b[i])
      .add(+1.0)
      .relu();
  });
  let data_loss = losses.reduce(
    (sum, current) => sum.add(current),
    new Value(0),
  ).mul(1/losses.length);
  // L2 regularization
  const alpha = 0.0001;
  let reg_loss = model
    .parameters()
    .map(e => e.mul(e))
    .reduce((sum, cur) => sum.add(cur), new Value(0))
    .mul(alpha);
  let total_loss = data_loss.add(reg_loss);

  const accuracies = range(0, scores.length).map(function (i) {
    return ((scores[i][0].data > 0.0) === (y_b[i] > 0.0)) ? 1.0 : 0.0;
  });
  const accuracy =
    accuracies.reduce((sum, current) => sum + current, 0.0) / accuracies.length;

  return [data_loss, reg_loss, total_loss, accuracy];
}

function do_epoch(k: number, model: any, X: any, y: any) {
  // # forward
  let [data_loss, reg_loss, total_loss, acc] = loss(X, y, model);

  // # backward
  model.zero_grad();
  total_loss.backward();

  // # update (sgd)
  // let learning_rate = 1.0 - (0.9 * k) / 100;
  let learning_rate = 0.001;
  for (let p of model.parameters()) {
    p.data -= learning_rate * p.grad;
  }

  if (k % 1 === 0) {
    console.log(
      `step ${k} loss ${total_loss.data}, data loss ${
        data_loss.data
      }, reg_loss ${reg_loss.data}, accuracy ${acc * 100}%`,
    );
  }
}

function train(nbEpochs, model, X, y) {
  for (let k of range(0, nbEpochs)) {
    do_epoch(k, model, X, y);
  }
}
