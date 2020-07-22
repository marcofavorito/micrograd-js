(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.microgradjs = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Value = exports.ensureValue = void 0;
function isNumber(x) {
    return typeof x === 'number';
}
function isString(x) {
    return typeof x === 'string';
}
function isValue(x) {
    return x instanceof Value;
}
function ensureValue(x) {
    if (isNumber(x)) {
        return new Value(x);
    }
    else {
        return x;
    }
}
exports.ensureValue = ensureValue;
var Value = /** @class */ (function () {
    function Value(data, children, op) {
        if (children === void 0) { children = []; }
        if (op === void 0) { op = ''; }
        this.data = data;
        this.children = children;
        this.op = op;
        this._backward = function () {
            return null;
        };
        this.grad = 0.0;
    }
    Value.prototype.add = function (other_) {
        var self = this;
        var other = ensureValue(other_);
        var out = new Value(this.data + other.data, [this, other], '+');
        out._backward = function () {
            self.grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    };
    Value.prototype.mul = function (other_) {
        var other = ensureValue(other_);
        var self = this;
        var out = new Value(this.data * other.data, [this, other], '*');
        out._backward = function () {
            self.grad += other.data * out.grad;
            other.grad += self.data * out.grad;
        };
        return out;
    };
    Value.prototype.pow = function (other_) {
        var other = other_;
        var self = this;
        var out = new Value(Math.pow(this.data, other), [this], '**' + other.toString());
        out._backward = function () {
            self.grad += other * Math.pow(self.data, other - 1) * out.grad;
        };
        return out;
    };
    Value.prototype.relu = function () {
        var self = this;
        var out = new Value(this.data < 0 ? 0.0 : this.data, [this], 'ReLU');
        out._backward = function () {
            self.grad += (out.data > 0.0 ? 1.0 : 0.0) * out.grad;
        };
        return out;
    };
    Value.prototype.backward = function () {
        var topo = [];
        var visited = new Set();
        var build_topo = function (v) {
            if (!visited.has(v)) {
                visited.add(v);
                for (var _i = 0, _a = v.children; _i < _a.length; _i++) {
                    var child = _a[_i];
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
            .forEach(function (v) {
            v._backward();
        });
    };
    Value.prototype.toString = function () {
        return "Value(data=" + this.data + ", grad=" + this.grad + ", op=" + this.op + ")";
    };
    return Value;
}());
exports.Value = Value;

},{}],2:[function(require,module,exports){
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !exports.hasOwnProperty(p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
__exportStar(require("./engine"), exports);
__exportStar(require("./nn"), exports);
__exportStar(require("./utils"), exports);

},{"./engine":1,"./nn":3,"./utils":4}],3:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.MLP = exports.Layer = exports.Neuron = exports.Module = void 0;
var engine_1 = require("./engine");
var utils_1 = require("./utils");
var Module = /** @class */ (function () {
    function Module() {
    }
    Module.prototype.parameters = function () {
        return [];
    };
    Module.prototype.call_value_ = function (x) {
        return [new engine_1.Value(0.0)];
    };
    Module.prototype.call_number_ = function (x) {
        var input = Array.from(utils_1.range(0, x.length), function (i) {
            return engine_1.ensureValue(x[i]);
        });
        return this.call_value_(input);
    };
    Module.prototype.call = function (x) {
        if (typeof x[0] === 'number') {
            return this.call_number_(x);
        }
        else {
            return this.call_value_(x);
        }
    };
    Module.prototype.zero_grad = function () {
        this.parameters().forEach(function (v) {
            v.grad = 0;
        });
    };
    return Module;
}());
exports.Module = Module;
var Neuron = /** @class */ (function (_super) {
    __extends(Neuron, _super);
    function Neuron(nin, nonlin) {
        if (nonlin === void 0) { nonlin = true; }
        var _this = _super.call(this) || this;
        _this.w = Array.from(utils_1.range(0, nin), function (x) { return new engine_1.Value(Math.random() * 2.0 - 1.0); });
        _this.b = new engine_1.Value(0.0);
        _this.nonlin = nonlin;
        return _this;
    }
    Neuron.prototype.call_value_ = function (x) {
        if (x.length != this.w.length) {
            throw new Error('Different sizes');
        }
        var act = this.w
            .map(function (e, i) {
            return e.mul(x[i]);
        })
            .reduce(function (sum, current) { return sum.add(current); }, new engine_1.Value(0.0))
            .add(this.b);
        return this.nonlin ? [act.relu()] : [act];
    };
    Neuron.prototype.parameters = function () {
        return this.w.concat([this.b]);
    };
    Neuron.prototype.toString = function () {
        return "${this.nonlin? 'ReLU': 'Linear'}Neuron(${this.w.length})";
    };
    return Neuron;
}(Module));
exports.Neuron = Neuron;
var Layer = /** @class */ (function (_super) {
    __extends(Layer, _super);
    function Layer(nin, nout, nonlin) {
        if (nonlin === void 0) { nonlin = true; }
        var _this = _super.call(this) || this;
        _this.neurons = Array.from(utils_1.range(0, nout), function (x) { return new Neuron(nin, nonlin); });
        return _this;
    }
    Layer.prototype.call_value_ = function (x) {
        var output = Array.from(this.neurons, function (n) { return n.call(x)[0]; });
        return output;
    };
    Layer.prototype.parameters = function () {
        var result = [];
        for (var _i = 0, _a = this.neurons; _i < _a.length; _i++) {
            var neuron = _a[_i];
            for (var _b = 0, _c = neuron.parameters(); _b < _c.length; _b++) {
                var param = _c[_b];
                result.push(param);
            }
        }
        return result;
    };
    Layer.prototype.toString = function () {
        return 'Layer of'; //TODO
    };
    return Layer;
}(Module));
exports.Layer = Layer;
var MLP = /** @class */ (function (_super) {
    __extends(MLP, _super);
    function MLP(nin, nouts) {
        var _this = _super.call(this) || this;
        var sizes = [nin].concat(nouts);
        _this.layers = Array.from(nouts.keys(), function (i) { return new Layer(sizes[i], sizes[i + 1], i != nouts.length - 1); });
        return _this;
    }
    MLP.prototype.call_value_ = function (x) {
        var result = this.layers[0].call(x);
        for (var i = 1; i < this.layers.length; i++) {
            result = this.layers[i].call(result);
        }
        return result;
    };
    MLP.prototype.parameters = function () {
        var result = [];
        for (var _i = 0, _a = this.layers; _i < _a.length; _i++) {
            var layer = _a[_i];
            for (var _b = 0, _c = layer.parameters(); _b < _c.length; _b++) {
                var param = _c[_b];
                result.push(param);
            }
        }
        return result;
    };
    MLP.prototype.toString = function () {
        return 'MLP of'; //TODO
    };
    return MLP;
}(Module));
exports.MLP = MLP;

},{"./engine":1,"./utils":4}],4:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.range = void 0;
function range(start, end) {
    return Array.from(Array(end - start).keys()).map(function (v) { return start + v; });
}
exports.range = range;

},{}]},{},[2])(2)
});
