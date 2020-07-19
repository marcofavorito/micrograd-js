function unif(low, high) {
  return Math.random() * (high - low) + low;
}

function moons_data(n) {
  var data = [];
  var labels = [];
  for (var i = 0; i < Math.PI; i += (Math.PI * 2) / n) {
    var point_1 = [
      Math.cos(i) + unif(-0.1, 0.1),
      Math.sin(i) + unif(-0.1, 0.1),
    ];
    data.push(point_1);
    labels.push(-1);

    var point_2 = [
      1 - Math.cos(i) + unif(-0.1, 0.1),
      1 - Math.sin(i) + unif(-0.1, 0.1) - 0.5,
    ];
    data.push(point_2);
    labels.push(1);
  }
  return [data, labels];
}

function circle_data(n) {
  var data = [];
  var labels = [];
  for (var i = 0; i < n / 2; i++) {
    var r = Math.random() * 2;
    var t = Math.random() * 2 * Math.PI;
    data.push([r * Math.sin(t), r * Math.cos(t)]);
    labels.push(1);
  }
  for (var i = 0; i < n / 2; i++) {
    var r = Math.random() * 2.0 + 3.0;
    var t = (2 * Math.PI * i) / 50.0;
    data.push([r * Math.sin(t), r * Math.cos(t)]);
    labels.push(-1);
  }
  return [data, labels];
}

function spiral_data(n) {
  var data = [];
  var labels = [];
  for (var i = 0; i < n / 2; i++) {
    var r = (i / n) * 5 + (Math.random() * 0.1 - 0.2);
    var t = ((1.25 * i) / n) * 2 * Math.PI + (Math.random() * 0.1 - 0.2);
    data.push([r * Math.sin(t), r * Math.cos(t)]);
    labels.push(1);
  }
  for (var i = 0; i < n / 2; i++) {
    var r = (i / n) * 5 + (Math.random() * 0.1 - 0.2);
    var t =
      ((1.25 * i) / n) * 2 * Math.PI + Math.PI + (Math.random() * 0.1 - 0.2);
    data.push([r * Math.sin(t), r * Math.cos(t)]);
    labels.push(-1);
  }
  return [data, labels];
}
