"""Download MNIST (the standard Keras mnist.npz) into data/mnist.npz and verify its SHA-256.

    python scripts/fetch_mnist.py

The file is not committed (it is a third-party dataset). The hash below is the one the reported results were run with.
"""
import hashlib
import os
import sys
import urllib.request

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist.npz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, expected=SHA256):
    got = sha256(path)
    if got != expected:
        raise ValueError(f"{path}: SHA-256 {got} does not match the expected {expected}")


def main():
    if os.path.exists(DEST):
        verify(DEST)
        print(f"{DEST} already present and verified")
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    tmp = DEST + ".part"
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, tmp)
    try:
        verify(tmp)
    except ValueError:
        os.remove(tmp)
        raise
    os.replace(tmp, DEST)
    print(f"saved and verified {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
