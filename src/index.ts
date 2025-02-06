import * as core from "./core";
import * as numpy from "./numpy";
import { Array, ArrayLike } from "./numpy";
import * as tree from "./tree";

export { numpy, tree };

// Fudged array types for composable transformations.

// export const jvpV1 = core.jvpV1 as unknown as (
//   f: (x: ArrayLike) => ArrayLike,
//   primals: ArrayLike[],
//   tangents: ArrayLike[]
// ) => [Array, Array];
export const deriv = core.deriv as unknown as (
  f: (x: ArrayLike) => ArrayLike
) => (x: ArrayLike) => Array;
