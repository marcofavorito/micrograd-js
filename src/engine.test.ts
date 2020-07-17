import Value from './engine';

test('test initialization', () => {
  const valueObj = new Value(3.14);
  expect(valueObj.data).toEqual(3.14);
});

test('test grad', () => {
  const v1 = new Value(2);
  const v2 = new Value(3);

  const plus = v1.add(v2);

  plus.backward();

  expect(plus.grad).toEqual(1);
  expect(v1.grad).toEqual(1);
  expect(v2.grad).toEqual(1);
});
