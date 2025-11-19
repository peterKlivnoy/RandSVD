# Quick Start Guide: Fixing Your Hadamard Transform

## The Problem
Your `hadamardKernel.py` file tries to import `_hadamardKernel` (a compiled C library), but it doesn't exist. This makes your SRHT implementation fall back to the slow NumPy version.

## The Solution (Pick One)

### 🎯 **RECOMMENDED: Install SciPy (2 minutes)**

This is the fastest and easiest solution:

```bash
pip install scipy
```

**That's it!** Your code already has fallback logic to use SciPy's fast Hadamard transform.

To verify it works:
```bash
python3 tests/test_fwht_implementation.py
```

You should see: `✅ SciPy's fht found`

---

### 🔧 **ADVANCED: Build Your Own C Library**

If you want the C library approach (for learning or maximum performance):

#### You're Missing These Files:
1. **C source code** (`.c` files with the actual FWHT implementation)
2. **Compiled library** (`_hadamardKernel.so` or `.pyd`)
3. **Build configuration** (`setup.py`)

#### Where Did You Get `hadamardKernel.py`?
The `hadamardKernel.py` file in your `src/` folder is a SWIG-generated wrapper. It came from somewhere - possibly:
- A library or package you partially installed?
- Example code from a tutorial?
- Files you copied from another project?

You need the **source** files that go with it!

#### If You Have the Source Files:
1. Place the `.c` files in your project
2. Create a `setup.py` (see `examples/setup_example.py`)
3. Build with: `python setup.py build_ext --inplace`
4. Move the resulting `.so`/`.pyd` file to your `src/` directory

#### If You Don't Have the Source Files:
I've created example C code in `examples/hadamard_example.c` that you can use as a starting point. However, **using SciPy is much easier and faster for a beginner**.

---

## Testing Your Setup

Run this to see which implementation you're using:
```bash
python3 tests/test_fwht_implementation.py
```

Then run your benchmark:
```bash
python3 tests/run_benchmark_1c_srht_speed.py
```

---

## Performance Comparison

| Implementation | Complexity | Speed | Setup Difficulty |
|---------------|-----------|-------|------------------|
| NumPy fallback | O(N²·N) | ⚠️ Slow | Already there |
| SciPy fht | O(N²·log N) | ✅ Fast | `pip install` |
| Custom C | O(N²·log N) | ✅✅ Fastest | ⚠️⚠️ Complex |

**For your use case, SciPy is perfect!**

---

## Next Steps

1. **Install SciPy**: `pip install scipy`
2. **Test**: `python3 tests/test_fwht_implementation.py`
3. **Run benchmarks**: `python3 tests/run_benchmark_1c_srht_speed.py`
4. **Celebrate**: Your code will now run much faster! 🎉

---

## Still Have Issues?

If you get errors, run the test script and share the output:
```bash
python3 tests/test_fwht_implementation.py
```
