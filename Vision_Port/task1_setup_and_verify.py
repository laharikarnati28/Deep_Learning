
"""
VisionPort - Task 1
Environment Setup and Library Verification

Expected Output:
Working Python Environment
"""

import sys
import importlib

print("=" * 60)
print("VISIONPORT - TASK 1: ENVIRONMENT SETUP")
print("=" * 60)

# 1. Check Python version
print("\n1. Checking Python Version")

print("Python Version:", sys.version)

if sys.version_info >= (3, 8):
    print("[PASSED] Python version is compatible")
else:
    print("[FAILED] Python 3.8 or higher is required")

# 2. Required libraries
required_libraries = [
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
    ("ultralytics", "YOLOv8"),
    ("torch", "PyTorch"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("multipart", "Python Multipart")
]

print("\n2. Checking Required Libraries")
print("-" * 60)

all_passed = True

for module_name, display_name in required_libraries:

    try:
        module = importlib.import_module(module_name)

        version = getattr(module, "__version__", "Installed")

        print(f"[PASSED] {display_name:<20} {version}")

    except ImportError:
        print(f"[MISSING] {display_name:<20} Not installed")
        all_passed = False

# 3. Final result
print("\n" + "=" * 60)
print("TASK 1 FINAL RESULT")
print("=" * 60)

if sys.version_info >= (3, 8) and all_passed:
    print("SUCCESS: Working Python Environment!")
    print("All required libraries are installed.")
else:
    print("SETUP INCOMPLETE: Install missing requirements.")

print("=" * 60)
