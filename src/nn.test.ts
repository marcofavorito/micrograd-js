import Value from './engine';
import { MLP } from './nn';

test('test MLP', () => {
  const nn = new MLP(2, [2, 2]);
  const output = nn.call([1, 2]);
  expect(output.length === 2);
});

test('test MLP with zero input', () => {
  const nn = new MLP(2, [32, 32, 32, 32, 2]);
  const output = nn.call([0, 0]);
  expect(output.length === 2);
  expect(output === [new Value(0.0), new Value(0.0)]);
});
