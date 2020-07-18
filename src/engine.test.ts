import { Value, ensureValue } from './engine';

test('test initialization', () => {
  const valueObj: Value = new Value(3.14);
  expect(valueObj.data).toEqual(3.14);

  const x: Value = ensureValue(valueObj);
  expect(x.data).toEqual(3.14);

  const y: Value = ensureValue(ensureValue(3.14));
  expect(y.data).toEqual(3.14);
});

test('test grad', () => {
  const v1: Value = new Value(2);
  const v2: Value = new Value(3);

  const plus: Value = v1.add(v2);

  plus.backward();

  expect(plus.grad).toEqual(1);
  expect(v1.grad).toEqual(1);
  expect(v2.grad).toEqual(1);
});
