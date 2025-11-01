# me - this DAT
# scriptOp - the OP which is cooking
import numpy as np

def onCook(scriptOp):
    # Get the parent component and extension
    ext = parent.stylegan
    
    # Try to import torch - if it fails, we're not ready yet
    try:
        import torch
    except ImportError as e:
        # Show initialization state and return early
        _show_initialization_state(scriptOp, f"Waiting for torch: {e}")
        return
    
    try:
        # Just try to generate - let it fail and recover naturally
        img_torch = ext.CookOutput()
        
        if img_torch is not None:
            height, width, channels = img_torch.shape
            address_int = img_torch.data_ptr()  # This is a Python int representing the GPU pointer
            num_bytes = img_torch.numel() * img_torch.element_size()  # total elements * bytes per element
            
            # Fill out shape descriptor
            shape = CUDAMemoryShape()
            shape.width = width
            shape.height = height
            shape.numComps = channels              # e.g. 4 for RGBA
            shape.dataType = np.uint8              # or 'numpy.uint8'
            
            # In TouchDesigner, scriptTOP.copyCUDAMemory(...) does a GPU-GPU copy if everything matches up
            scriptOp.copyCUDAMemory(address_int, num_bytes, shape)
        else:
            # If no image was generated, create a blank image
            width = parent().par.Width.eval() if parent().par.Customsize else 1024
            height = parent().par.Height.eval() if parent().par.Customsize else 1024
            blank = np.zeros((height, width, 4), dtype=np.uint8)
            blank[:,:,3] = 255  # Set alpha to 255
            scriptOp.copyNumpyArray(blank)
    
    except Exception as e:
        # In case of any error, provide a small error indicator image
        error_img = np.zeros((256, 256, 4), dtype=np.uint8)
        error_img[:,:,0] = 255  # Red color to indicate error
        error_img[:,:,3] = 255  # Alpha
        scriptOp.copyNumpyArray(error_img)
        print(f"Error in StyleGAN scriptTOP: {e}")
        import traceback
        traceback.print_exc()

def _show_initialization_state(scriptOp, message):
    """
    Show a blue initialization/waiting state image with status message
    """
    # Create a blue image to indicate initialization state
    init_img = np.zeros((256, 256, 4), dtype=np.uint8)
    init_img[:,:,2] = 128  # Blue color for initialization
    init_img[:,:,3] = 255  # Alpha
    scriptOp.copyNumpyArray(init_img)
    print(f"StyleGAN initializing: {message}")