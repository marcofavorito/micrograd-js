function isNumber(x: any): x is number {
  return typeof x === 'number';
}

function isString(x: any): x is string {
  return typeof x === 'string';
}

function isValue(x: any): x is Value {
  return x instanceof Value;
}

export function ensureValue(x: number | Value): Value {
  if (isNumber(x)) {
    return new Value(x);
  } else {
    return x;
  }
}

export class Value {
  data: number;
  children: Value[];
  op: string;
  _backward: () => void;
  grad: number;
  constructor(data: number, children: Value[] = [], op = '') {
    this.data = data;
    this.children = children;
    this.op = op;

    this._backward = function () {
      return null;
    };
    this.grad = 0.0;
  }

  add(other_: Value | number): Value {
    const self = this;
    const other: Value = ensureValue(other_);
    const out: Value = new Value(this.data + other.data, [this, other], '+');

    out._backward = function () {
      self.grad += out.grad;
      other.grad += out.grad;
    };

    return out;
  }

  mul(other_: Value | number): Value {
    const other: Value = ensureValue(other_);
    const self = this;
    const out: Value = new Value(this.data * other.data, [this, other], '*');

    out._backward = function () {
      self.grad += other.data * out.grad;
      other.grad += self.data * out.grad;
    };

    return out;
  }

  pow(other_: number): Value {
    const other = other_;
    const self = this;
    const out = new Value(
      Math.pow(this.data, other),
      [this],
      '**' + other.toString(),
    );

    out._backward = function () {
      self.grad += other * Math.pow(self.data, other - 1) * out.grad;
    };

    return out;
  }

  relu(): Value {
    const self = this;
    const out = new Value(this.data < 0 ? 0.0 : this.data, [this], 'ReLU');

    out._backward = function () {
      self.grad += (out.data > 0.0 ? 1.0 : 0.0) * out.grad;
    };
    return out;
  }

  backward(): void {
    const topo = [];
    const visited = new Set();
    const build_topo = function (v: Value) {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v.children) {
          build_topo(child);
        }
        topo.push(v);
      }
    };
    build_topo(this);

    this.grad = 1;
    topo
      .slice()
      .reverse()
      .forEach(function (v: Value) {
        v._backward();
      });
  }

  toString(): string {
    return `Value(data=${this.data}, grad=${this.grad}, op=${this.op})`;
  }
}
