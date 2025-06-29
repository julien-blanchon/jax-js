import { devices, grad, init, nn, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  suite("jax.nn.relu()", () => {
    test("should compute ReLU", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.relu(x);
      expect(y.js()).toEqual([0, 0, 1, 2]);
    });

    test("should compute ReLU gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.relu(x).sum());
      const gx = gradFn(x);
      expect(gx.js()).toEqual([0, 0, 1, 1]);
    });
  });
});
