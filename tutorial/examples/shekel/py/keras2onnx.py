#!/usr/bin/python3.12
import tensorflow as tf
import tf2onnx
import onnx
import os
from sys import argv
import hashlib

original_rename = tf2onnx.convert._rename_duplicate_keras_model_names

def calculate_file_checksum(file_path, algorithm='sha256'):
    """
    Compute the hash of a file using the specified algorithm (e.g., 'md5', 'sha1', 'sha256').
    """
    # Create a hash object for the specified algorithm
    hash_func = hashlib.new(algorithm)
    
    # Open the file in binary mode ('rb')
    try:
        with open(file_path, 'rb') as f:
            # Read the file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError:
        print(f"Error: Invalid hash algorithm '{algorithm}'")
        return None

def patched_rename(model):
    # Add output_names if it doesn't exist
    if not hasattr(model, 'output_names'):
        model.output_names = [output.name.split(':')[0] for output in model.outputs]
    return original_rename(model)

def main(rootpath: str = ".") -> int:
    os.environ['CUDA_VISIBLE_DEVICES']='-1'

    tf2onnx.convert._rename_duplicate_keras_model_names = patched_rename
    # Load your Keras model
    model = tf.keras.models.load_model(f'{rootpath}/shekel_model_expected.keras')

    # Convert directly to ONNX - tf2onnx should handle this automatically
    spec = tf.TensorSpec(model.input_shape, tf.float32, name="input")
    onnx_model, _ = tf2onnx.convert.from_keras(model, [spec], opset=13)

    # Save the ONNX model
    onnx_file_path=f'{rootpath}/shekel_model_expected.onnx'
    onnx.save(onnx_model, onnx_file_path)
    return calculate_file_checksum(onnx_file_path, algorithm='sha256')

if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    print(main(rootpath))
