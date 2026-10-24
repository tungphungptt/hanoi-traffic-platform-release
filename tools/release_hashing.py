from pathlib import Path

BINARY_SUFFIXES = {
    ".onnx", ".pth", ".pt", ".pkl", ".pickle", ".safetensors",
    ".bin", ".npz", ".npy", ".h5", ".hdf5", ".tfrecord",
    ".parquet", ".arrow", ".feather", ".mat",
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg",
    ".m4v", ".mts", ".m2ts", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".ico", ".pdf", ".zip", ".gz", ".woff", ".woff2",
    ".ttf", ".otf", ".sqlite", ".db", ".exe", ".dll", ".so",
    ".dylib", ".wasm", ".tflite", ".engine", ".weights", ".ckpt",
}

def canonical_release_bytes(path: Path, data: bytes) -> bytes:
    if path.suffix.lower() in BINARY_SUFFIXES or b"\x00" in data:
        return data
    return data.replace(b"\r\n", b"\n")