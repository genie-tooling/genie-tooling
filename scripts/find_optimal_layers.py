#!/usr/bin/env python3
"""
Developer Utility: Find Optimal GPU Layers for a GGUF Model

This script determines the maximum number of layers of a given GGUF model that can be
offloaded to the GPU on the current hardware. It uses a binary search algorithm
for efficiency.

The output provides the optimal 'n_gpu_layers' value to use in your application's
configuration for the best performance.

This script is designed to be compatible with all modern versions of `llama-cpp-python`.

Usage:
    poetry run python scripts/find_optimal_layers.py --model-path /path/to/your/model.gguf

Prerequisites:
- `genie-tooling` must be installed with the `llama_cpp_internal` extra.
  `poetry install --extras llama_cpp_internal`
- The `llama-cpp-python` library must be compiled with GPU support (e.g., CUDA, Metal).
"""
import argparse
import logging
import sys
from pathlib import Path

# Configure logging to be minimal for this utility
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: 'llama-cpp-python' is not installed. Please install it to use this utility:")
    print("  poetry install --extras llama_cpp_internal")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to import Llama from llama-cpp-python: {e}")
    print("Please ensure your installation is correct.")
    sys.exit(1)


def get_total_layers(model_path: str) -> int:
    """
    Gets the total number of layers from a GGUF model by creating a temporary
    CPU-only instance to inspect its parameters.
    """
    print("INFO: Reading model metadata... (This may take a moment as it loads the model structure)")
    try:
        # Load with 0 GPU layers just to inspect the metadata without using VRAM
        llm = Llama(model_path=model_path, n_gpu_layers=0, verbose=False)
        # --- FINAL CORRECTED METHOD ---
        # Access the n_layer property directly on the Llama object.
        total_layers = llm.n_layer
        del llm # Immediately release the object and its resources
        return total_layers
    except Exception as e:
        raise RuntimeError(f"Failed to get layer count by loading model on CPU: {e}") from e


def find_optimal_gpu_layers(model_path: str, total_layers: int) -> int:
    """
    Performs a binary search to find the maximum number of GPU layers that can be loaded.

    Args:
        model_path: The path to the GGUF model file.
        total_layers: The total number of layers in the model.

    Returns:
        The optimal number of GPU layers. Returns 0 if loading fails even with 0 layers.
    """
    print("\nStarting binary search for optimal n_gpu_layers...")

    low = 0
    high = total_layers
    optimal_layers = 0

    # First, check if even 1 layer can be offloaded. If not, no point in searching.
    print("  - Probing with 1 layer (initial check)...", end="", flush=True)
    try:
        llm_check = Llama(model_path=model_path, n_gpu_layers=1, n_ctx=512, verbose=False)
        del llm_check
        print(" SUCCESS")
    except Exception:
        print(" FAILED")
        print("INFO: Could not offload even one layer to the GPU. Ensure llama-cpp-python was compiled with GPU support.")
        return 0


    while low <= high:
        mid = (low + high) // 2
        if mid == 0: # We already know 0 works, and we've tested 1. Start search from 1.
            low = 1
            continue

        print(f"  - Probing with {mid} layers...", end="", flush=True)
        try:
            # Attempt to initialize the model with the current number of layers.
            llm = Llama(model_path=model_path, n_gpu_layers=mid, n_ctx=512, verbose=False)
            del llm  # Release the context immediately
            print(" SUCCESS")
            # If successful, this is a potential answer. Try for more layers.
            optimal_layers = mid
            low = mid + 1
        except ValueError as e:
            # This is the expected error for VRAM exhaustion
            if "Failed to create llama_context" in str(e):
                print(" FAILED (VRAM Exceeded)")
                # If failed, this number is too high. Try with fewer layers.
                high = mid - 1
            else:
                print(f" FAILED (Unexpected ValueError: {e})")
                # For other errors, it might not be a VRAM issue. Stop searching.
                high = mid - 1
        except Exception as e:
            print(f" FAILED (Critical Error: {e})")
            # For critical errors, stop the search.
            break

    return optimal_layers


def main():
    """Main function to parse arguments and run the search."""
    parser = argparse.ArgumentParser(
        description="Find the optimal n_gpu_layers for a GGUF model on your hardware.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to the GGUF model file."
    )
    args = parser.parse_args()

    model_file = Path(args.model_path)
    if not model_file.exists() or not model_file.is_file():
        print(f"ERROR: Model file not found at '{args.model_path}'")
        sys.exit(1)

    print("-" * 60)
    print("Genie Tooling - Optimal GPU Layer Finder")
    print("-" * 60)
    print(f"Analyzing model: {model_file.name}")

    try:
        total_layers = get_total_layers(args.model_path)
        print(f"Model has a total of {total_layers} layers.")
    except Exception as e:
        print("\nFATAL ERROR: Could not determine the number of layers in the model.")
        print(f"Reason: {e}")
        sys.exit(1)

    optimal_n_gpu_layers = find_optimal_gpu_layers(args.model_path, total_layers)

    print("\n" + "=" * 60)
    print("✨ Search Complete ✨")
    print(f"Optimal n_gpu_layers found: {optimal_n_gpu_layers}")
    print("=" * 60)

    print("\nRecommended `FeatureSettings` configuration for this model:")
    print("-" * 60)
    print("from genie_tooling.config import FeatureSettings")
    print("")
    print("features = FeatureSettings(")
    print("    llm='llama_cpp_internal',")
    print(f"    llm_llama_cpp_internal_model_path='{model_file.resolve()}',")
    print(f"    llm_llama_cpp_internal_n_gpu_layers={optimal_n_gpu_layers},")
    print("    # ... other settings like n_ctx")
    print(")")
    print("-" * 60)


if __name__ == "__main__":
    main()
