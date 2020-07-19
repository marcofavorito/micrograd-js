import { Value, ensureValue } from './engine';
import { range } from './utils';

export class Module {
  parameters(): Value[] {
    return [];
  }
  call_value_(x: Value[]): Value[] {
    return [new Value(0.0)];
  }
  call_number_(x: number[]): Value[] {
    const input: Value[] = Array.from(range(0, x.length), i =>
      ensureValue(x[i]),
    );
    return this.call_value_(input);
  }
  call(x: Value[] | number[]): Value[] {
    if (typeof x[0] === 'number') {
      return this.call_number_(<number[]>x);
    } else {
      return this.call_value_(<Value[]>x);
    }
  }

  zero_grad(): void {
    this.parameters().forEach(function (v: Value) {
      v.grad = 0;
    });
  }
}

export class Neuron extends Module {
  w: Value[];
  b: Value;
  nonlin: boolean;

  constructor(nin, nonlin = true) {
    super();
    this.w = Array.from(
      range(0, nin),
      x => new Value(Math.random() * 2.0 - 1.0),
    );
    this.b = new Value(0.0);
    this.nonlin = nonlin;
  }

  call_value_(x: Value[]): Value[] {
    if (x.length != this.w.length) {
      throw new Error('Different sizes');
    }
    const act = this.w
      .map(function (e: Value, i: number): Value {
        return e.mul(x[i]);
      })
      .reduce((sum, current) => sum.add(current), new Value(0.0))
      .add(this.b);
    return this.nonlin ? [act.relu()] : [act];
  }

  parameters(): Value[] {
    return this.w.concat([this.b]);
  }

  toString(): string {
    return "${this.nonlin? 'ReLU': 'Linear'}Neuron(${this.w.length})";
  }
}

export class Layer extends Module {
  neurons: Neuron[];

  constructor(nin: number, nout: number, nonlin = true) {
    super();
    this.neurons = Array.from(range(0, nout), x => new Neuron(nin, nonlin));
  }

  call_value_(x: Value[]): Value[] {
    const output = Array.from(this.neurons, n => n.call(x)[0]);
    return output;
  }

  parameters(): Value[] {
    const result: Value[] = [];
    for (const neuron of this.neurons) {
      for (const param of neuron.parameters()) {
        result.push(param);
      }
    }
    return result;
  }

  toString(): string {
    return 'Layer of'; //TODO
  }
}

export class MLP extends Module {
  layers: Layer[];
  constructor(nin: number, nouts: number[]) {
    super();
    const sizes = [nin].concat(nouts);
    this.layers = Array.from(
      nouts.keys(),
      i => new Layer(sizes[i], sizes[i + 1], i != nouts.length - 1),
    );
  }

  call_value_(x: Value[]): Value[] {
    let result: Value[] = this.layers[0].call(x);
    for (let i = 1; i < this.layers.length; i++) {
      result = this.layers[i].call(result);
    }
    return result;
  }

  parameters(): Value[] {
    const result: Value[] = [];
    for (const layer of this.layers) {
      for (const param of layer.parameters()) {
        result.push(param);
      }
    }
    return result;
  }

  toString() {
    return 'MLP of'; //TODO
  }
}
