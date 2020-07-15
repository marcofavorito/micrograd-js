export default class Value {
  data: number;
  children: Value[];
  op: string;
  //_backward: () => number;
  constructor(data: number, children: Value[] = [], op = '') {
    this.data = data;
    this.children = children;
    this.op = op;

    //this._backward = () => {null;}
  }
}
