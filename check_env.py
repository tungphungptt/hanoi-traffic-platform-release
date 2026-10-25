import sys
import platform
import subprocess
import shutil

print("=== Python and OS ===")
print("Python:", sys.version)
print("Executable:", sys.executable)
print("Platform:", platform.platform())

packages = ['numpy', 'scipy', 'sklearn', 'cv2', 'onnxruntime', 'torch', 'torchvision']
print("\n=== Package Versions ===")
for p in packages:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', 'installed (no version attr)')
        print(f"{p}: {ver}")
    except ImportError:
        print(f"{p}: NOT INSTALLED")

print("\n=== GPU / Hardware ===")
try:
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    print("nvidia-smi returncode:", smi.returncode)
    if smi.returncode == 0:
        lines = smi.stdout.strip().split("\n")
        for line in lines[:15]:
            print(line)
except Exception as e:
    print("nvidia-smi not available:", e)

total, used, free = shutil.disk_usage(".")
print(f"\n=== Disk Space ===")
print(f"Total: {total / (1024**3):.2f} GB | Free: {free / (1024**3):.2f} GB")
