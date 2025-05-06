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
  const gidxVar = AluExp.special(DType.Int32, "gidx", kernel.size);
  let ridxVar: AluExp | undefined;
  if (kernel.reduction) {
    ridxVar = AluExp.special(DType.Int32, "ridx", kernel.reduction.size);
  }

  return {
    exp: lowerExp(kernel.exp, gidxVar, ridxVar).simplify(),
  };
}

/** Tuning for WebGPU kernels. */
export function tuneWebgpu(kernel: Kernel): TuneResult {
  // TODO: Implement WebGPU tuning.
  return tuneNullopt(kernel);
}

function lowerExp(exp: AluExp, gidxVar: AluExp, ridxVar?: AluExp): AluExp {
  let newSrc = exp.src.map((e) => lowerExp(e, gidxVar, ridxVar));
  if (
    newSrc.length === exp.src.length &&
    newSrc.every((s, i) => s === exp.src[i])
  ) {
    newSrc = exp.src; // No changes, so reuse the original.
  }

  if (exp.op === AluOp.GlobalView) {
    const gid: number = exp.arg[0];
    const st: ShapeTracker = exp.arg[1];
    return accessorGlobal(gid, st, newSrc);
  } else if (exp.op === AluOp.Variable) {
    if (exp.arg === "gidx") {
      return gidxVar;
    } else if (exp.arg === "ridx") {
      if (ridxVar) return ridxVar;
      else throw new Error("ridx variable not provided");
    }
    return exp;
  }

  if (newSrc !== exp.src) {
    return new AluExp(exp.op, exp.dtype, newSrc, exp.arg);
  }
  return exp;
}
