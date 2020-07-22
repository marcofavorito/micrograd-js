import { Value } from './engine';
import { Layer, MLP, Neuron } from './nn';
import exp = require('constants');

test('test MLP', () => {
  const nn = new MLP(2, [2, 2]);
  const output = nn.call([1, 2]);
  expect(output.length).toEqual(2);
});

test('test MLP with zero input', () => {
  const nn = new MLP(2, [32, 32, 32, 32, 2]);
  const output = nn.call([0, 0]);
  expect(output[0].data).toEqual(0.0);
  expect(output[1].data).toEqual(0.0);
});

test('test Layer forward', () => {
  const l = new Layer(2, 2);
  const x = [2.0, -2.0];
  const n1 = l.neurons[0];
  const n2 = l.neurons[1];
  n1.b.data = 100.0;
  n1.w[0].data = 2.0;
  n1.w[1].data = 3.0;

  n2.b.data = 200.0;
  n2.w[0].data = -5.0;
  n2.w[1].data = 10.0;

  const actual_out = l.call(x);
  const expected_out = [
    n1.w[0].data * x[0] + n1.w[1].data * x[1] + n1.b.data,
    n2.w[0].data * x[0] + n2.w[1].data * x[1] + n2.b.data,
  ];

  expect(actual_out[0].data).toEqual(expected_out[0]);
  expect(actual_out[1].data).toEqual(expected_out[1]);
});

test('test MLP training', () => {
  const nn = new MLP(2, [16, 16, 16, 1]);
  const losses = [];

  const x = [-1.0, 2.0];
  const y = 2.0;
  expect(nn.call(x)[0].data).not.toBeCloseTo(2.0, 2);

  for (let i = 0; i < 200; i++) {
    const out: Value[] = nn.call(x);
    const loss = out[0].add(-y).pow(2);
    losses.push(loss.data);

    nn.zero_grad();
    loss.backward();

    for (const p of nn.parameters()) {
      p.data -= 0.0001 * p.grad;
    }
  }
  expect(nn.call(x)[0].data).toBeCloseTo(2.0, 1);
});
