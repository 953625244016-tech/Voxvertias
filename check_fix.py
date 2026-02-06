import torch
try:
    import torch.optim as optim
    from torch.utils._pytree import _register_pytree_node
    print("✅ Success! Torch is now compatible.")
    
    # Check if the problematic import works now
    import torch.onnx
    print("✅ Success! ONNX system is healthy.")
except Exception as e:
    print(f"❌ Still failing: {e}")