"""
Quick test to check which FWHT implementation is being used
"""
import sys
import os

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

print("=" * 60)
print("Testing FWHT Implementation Detection")
print("=" * 60)

# This will print which implementation is detected
from src import structured_sketch

print("\nChecking what's available:")
print(f"  FAST_FWHT_AVAILABLE: {structured_sketch.FAST_FWHT_AVAILABLE}")
print(f"  fastwht_func: {structured_sketch.fastwht_func}")

# Test imports individually
print("\n" + "=" * 60)
print("Individual Package Check:")
print("=" * 60)

# Check hadamardKernel
try:
    import hadamardKernel
    print("✅ hadamardKernel module found")
    try:
        func = hadamardKernel.fwhtKernel2dOrdinary
        print("   ✅ fwhtKernel2dOrdinary function accessible")
    except AttributeError as e:
        print(f"   ❌ Function not accessible: {e}")
except ImportError as e:
    print(f"❌ hadamardKernel not found: {e}")

# Check scipy
try:
    from scipy.fft import fht
    print("✅ SciPy's fht found")
except ImportError:
    print("❌ SciPy's fht not found")

# Check if scipy is installed at all
try:
    import scipy
    print(f"✅ SciPy version {scipy.__version__} installed")
except ImportError:
    print("❌ SciPy not installed")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)

if structured_sketch.FAST_FWHT_AVAILABLE:
    print("✅ You have a fast implementation!")
    print("   Your benchmarks will run with O(N² log N) complexity.")
else:
    print("⚠️  No fast implementation detected!")
    print("   Install SciPy for a quick fix:")
    print("   ")
    print("   pip install scipy")
    print("   ")
    print("   Or see README_HADAMARD_SETUP.md for other options.")
