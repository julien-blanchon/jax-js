/**
 * @file Optimizations applied to kernels by different backends.
 *
 * The main optimizations (for reductions) are:
 *
 * - "Unroll": Multiple values or loop iterations are computed per thread.
 *   - Along reduce dimension: traditional loop unrolling, so you would
 *     increment the loop index by the unroll factor.
 *   - Along other dimension: each thread computes a block of output values,
 *     which helps with cache performance (e.g., matmul tiling).
 *
 * - "Group": Multiple threads compute the same value. For example, when summing
 *   up the numbers in a vector, K threads each accumulate 1/K of the vector,
 *   stores in shared memory, and thread 0 accumulates at the end.
 *   - Regular order: 4 threads grouped as [1234123412341234]
 *   - "Top": 4 threads grouped as [1111222233334444]
 *
 * - "Upcast": Similar to Unroll, but for vector/SIMD instructions.
 *
 * These are inspired by Tinygrad's heuristic optimizations.
 * https://github.com/tinygrad/tinygrad/blob/685d5c46df/tinygrad/codegen/heuristic.py
 */

import { accessorGlobal, AluExp, AluOp, DType, Kernel } from "./alu";
import { ShapeTracker } from "./shape";

// gidx = (0 ... dim.local ... dim.reduce)
// ridx = (dim.reduce .[local index].
//         dim.group .[reduce loops].
//         dim.unrollagg .[unroll]. dim.unroll)
// uidx = (dim.unrollagg .[unroll]. dim.unroll)
// result[gidx + uidx] = <<< eval(kernel.exp) >>>;

export interface TuneResult {
  /** New expression with GlobalView ops and gidx/ridx lowered. */
  exp: AluExp;

  /** Applied shape for optimizations to all arguments in the tuned kernel. */
  shape?: ShapeTracker;

  /** Dimensions of the kernel's applied shape. Globals start at 0. */
  dim?: {
    // local: number; // TODO: Split gidx -> global and local dimensions during tuning.
    reduce: number; // Reductions start here.
    group: number; // Single reduction thread, equal to reduce if no groups.
    unrollagg: number; // Unroll along the reduce dimension.
    unroll: number; // Unroll along output dimension.
  };
}

/** Tuning step that does not apply any optimization. */
export function tuneNullopt(kernel: Kernel): TuneResult {
  const vars: Record<string, AluExp> = {};
  vars.gidx = AluExp.special(DType.Int32, "gidx", kernel.size);
  if (kernel.reduction)
    vars.ridx = AluExp.special(DType.Int32, "ridx", kernel.reduction.size);
  return {
    exp: lowerExp(kernel.exp).substitute(vars).simplify(),
  };
}

/** Tuning for WebGPU kernels. */
export function tuneWebgpu(kernel: Kernel): TuneResult {
  if (!kernel.reduction) return tuneNullopt(kernel);
  // TODO: Implement WebGPU tuning.

  // 1. Check that kernel GlobalView ops have consistent src[], where the last
  //    dimension is reduction, and others are gidx.
  // 2. Collect all shape trackers for kernel GlobalView ops.
  // 3. Apply heuristic optimizations based on the shape trackers.
  // 4. Return the tuned kernel result.

  return tuneNullopt(kernel);
}

function lowerExp(exp: AluExp): AluExp {
  return exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const gid: number = exp.arg[0];
      const st: ShapeTracker = exp.arg[1];
      return accessorGlobal(gid, st, exp.src);
    }
  });
}
