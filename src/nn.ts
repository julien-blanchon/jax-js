// Common functions for neural network libraries, mirroring `jax.nn` in JAX.

import { Array, ArrayLike, maximum } from "./numpy";

/** ReLU activation function: `relu(x) = max(x, 0)`. */
export function relu(x: ArrayLike): Array {
  return maximum(x, 0);
}
