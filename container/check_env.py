"""Import every dependency and print its version. Never exits non-zero, so a broken optional import
does not fail a container build; read the output instead.

Run inside the container:  apptainer exec --nv $SIF python /opt/check_env.py
"""

import importlib
import platform

MODULES = [
    "torch", "torchvision", "numpy", "scipy", "pandas", "pyarrow", "PIL", "yaml", "pycocotools",
    "transformers", "huggingface_hub", "safetensors", "cv2", "sam3", "sam2",
]


def main() -> None:
    print("python", platform.python_version())
    for name in MODULES:
        try:
            m = importlib.import_module(name)
            print("OK  ", name, getattr(m, "__version__", ""))
        except Exception as e:
            print("FAIL", name, type(e).__name__, e)
    try:
        import torch

        print("cuda available:", torch.cuda.is_available(), "| torch built for CUDA", torch.version.cuda)
    except Exception as e:
        print("FAIL torch cuda check", e)


if __name__ == "__main__":
    main()
