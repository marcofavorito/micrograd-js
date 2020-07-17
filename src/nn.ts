import Value from './engine';
import ensureValue from './engine';
import { range } from './utils';

export class Module {
  parameters(): Value[] {
    return [];
  }
  call(x: any[]): Value[] {
    return [new Value(0.0)];
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

  call(x: any[]): Value[] {
    const input = Array.from(range(0, x.length), i => new ensureValue(x[i]));
    const act = this.w
      .map(function (e: Value, i: number): Value {
        return e.mul(input[i]);
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

  call(x: any[]): Value[] {
    const input = Array.from(range(0, x.length), i => new ensureValue(x[i]));
    const output = Array.from(this.neurons, n => n.call(input)[0]);
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
      sizes,
      i => new Layer(sizes[i], sizes[i + 1], i != nouts.length),
    );
  }

  call(x: any[]): Value[] {
    let result: Value[] = this.layers[0].call(x);
    for (const layer of this.layers.slice(1)) {
      result = layer.call(result);
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
