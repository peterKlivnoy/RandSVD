# Setting Up Fast Hadamard Transform C Library

## Current Issue
Your `hadamardKernel.py` file is trying to import `_hadamardKernel` (a compiled C library), but the library doesn't exist. This causes the code to fall back to the slow NumPy implementation.

## Solution Options (Choose ONE)

### **Option 1: Use SciPy (EASIEST - Recommended for beginners)**

SciPy already has a fast Hadamard transform built-in!

1. Install scipy if you haven't:
   ```bash
   pip install scipy
   ```

2. Your code is already set up to use SciPy as a fallback! Just make sure scipy is installed.

3. Test it:
   ```bash
   python3 tests/run_benchmark_1c_srht_speed.py
   ```

**Pros:**
- No compilation needed
- Works immediately
- Well-tested and maintained
- Fast O(N log N) implementation

**Cons:**
- Slightly slower than optimized C code (but still much faster than NumPy fallback)

---

### **Option 2: Use PyFFTW (Alternative Fast Library)**

PyFFTW is another package with fast transforms:

```bash
pip install pyfftw
```

Then modify `structured_sketch.py` to use it.

---

### **Option 3: Build Your Own C Library (ADVANCED)**

If you want to learn how to integrate C code into Python:

#### What You Need:
1. C source code for Fast Walsh-Hadamard Transform
2. SWIG interface file (`.i` file)
3. `setup.py` to compile the C code

#### Steps:

1. **Get the C source code**: You need a `.c` file with the FWHT implementation. Where did you get `hadamardKernel.py` from? You need the corresponding `.c` files.

2. **Create a `setup.py`** (I can help you with this once you have the C files)

3. **Compile**: Run `python setup.py build_ext --inplace`

4. **This will generate**: `_hadamardKernel.so` (on Mac/Linux) or `_hadamardKernel.pyd` (on Windows)

## Current Status
- ✅ You have: `hadamardKernel.py` (Python wrapper)
- ❌ You're missing: `_hadamardKernel.so` (compiled C library)
- ❌ You're missing: C source files (`.c` or `.cpp`)
- ❌ You're missing: `setup.py` (build configuration)

## Quick Fix (Use SciPy)

The fastest way to fix this NOW:

```bash
pip install scipy
```

Your code will automatically detect and use SciPy's fast Hadamard transform!
