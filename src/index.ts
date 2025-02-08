import * as core from "./core";
import * as numpy from "./numpy";
import { Array, ArrayLike } from "./numpy";
import * as tree from "./tree";
import type { JsTree } from "./tree";

export { numpy, tree };

type MapToArrayLike<T> = T extends Array
  ? ArrayLike
  : T extends globalThis.Array<infer U>
    ? MapToArrayLike<U>[]
    : { [K in keyof T]: MapToArrayLike<T[K]> };

type WithArgsSubtype<F extends (args: any[]) => any, T> =
  Parameters<F> extends T ? F : never;

export const jvp = core.jvp as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<Array>>,
  primals: MapToArrayLike<Parameters<F>>,
  tangents: MapToArrayLike<Parameters<F>>
) => [ReturnType<F>, ReturnType<F>];

export const deriv = core.deriv as unknown as (
  f: (x: ArrayLike) => ArrayLike
) => (x: ArrayLike) => Array;
