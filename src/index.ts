import * as core from "./core";
import * as numpy from "./numpy";
import { Array, ArrayLike } from "./numpy";
import * as tree from "./tree";

export { numpy, tree };

// TODO: Improve the type hints here.
export const jvp = core.jvp as <F extends (...args: any) => any>(
  f: F,
  primals: Parameters<F>,
  tangents: Parameters<F>
) => [ReturnType<F>, ReturnType<F>];

export const deriv = core.deriv as unknown as (
  f: (x: ArrayLike) => ArrayLike
) => (x: ArrayLike) => Array;
