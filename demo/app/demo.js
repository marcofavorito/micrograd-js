function loss(X, y, model, batch_size) {
    if (batch_size === void 0) { batch_size = null; }
    // TODO use batch_size
    var X_b = X;
    var y_b = y;
    var inputs = X_b.map(function (row) {
        return row.map(function (x) { return microgradjs.ensureValue(x); });
    });
    var labels = y_b.map(function (label) {
        return microgradjs.ensureValue(label);
    });
    var scores = inputs.map(function (row) { return model.call(row); });
    // svm max margin loss
    var losses = microgradjs.range(0, scores.length).map(function (i) {
        return scores[i][0]
            .mul(-y_b[i])
            .add(+1.0)
            .relu();
    });
    var data_loss = losses.reduce(function (sum, current) { return sum.add(current); }, new microgradjs.Value(0)).mul(1 / losses.length);
    // L2 regularization
    var alpha = 0.0001;
    var reg_loss = model
        .parameters()
        .map(function (e) { return e.mul(e); })
        .reduce(function (sum, cur) { return sum.add(cur); }, new microgradjs.Value(0))
        .mul(alpha);
    var total_loss = data_loss.add(reg_loss);
    var accuracies = microgradjs.range(0, scores.length).map(function (i) {
        return ((scores[i][0].data > 0.0) === (y_b[i] > 0.0)) ? 1.0 : 0.0;
    });
    var accuracy = accuracies.reduce(function (sum, current) { return sum + current; }, 0.0) / accuracies.length;
    return [data_loss, reg_loss, total_loss, accuracy];
}
function do_epoch(k, model, X, y) {
    // # forward
    var _a = loss(X, y, model), data_loss = _a[0], reg_loss = _a[1], total_loss = _a[2], acc = _a[3];
    // # backward
    model.zero_grad();
    total_loss.backward();
    // # update (sgd)
    // let learning_rate = 1.0 - (0.9 * k) / 100;
    var learning_rate = 0.001;
    for (var _i = 0, _b = model.parameters(); _i < _b.length; _i++) {
        var p = _b[_i];
        p.data -= learning_rate * p.grad;
    }
    if (k % 1 === 0) {
        console.log("step " + k + " loss " + total_loss.data + ", data loss " + data_loss.data + ", reg_loss " + reg_loss.data + ", accuracy " + acc * 100 + "%");
    }
}
function train(nbEpochs, model, X, y) {
    for (var _i = 0, _a = microgradjs.range(0, nbEpochs); _i < _a.length; _i++) {
        var k = _a[_i];
        do_epoch(k, model, X, y);
    }
}
