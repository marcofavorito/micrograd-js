import Value from './engine';

test('test initialization', () => {
  const valueObj = new Value(3.14);
  expect(valueObj.data).toEqual(3.14);
});
