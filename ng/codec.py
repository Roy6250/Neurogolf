import numpy as np

CHANNELS, HEIGHT, WIDTH = 10, 30, 30

def encode_grid(grid):
    """Encodes a 2D grid of integers (0-9) into a (1, 10, 30, 30) one-hot float32 tensor."""
    tensor = np.zeros((1, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color < 0 or color >= CHANNELS:
                raise ValueError(f"Invalid color: {color}")
            tensor[0, color, r, c] = 1.0
    return tensor

def decode_tensor(tensor):
    """Decodes a (1, 10, 30, 30) tensor back to a 2D list grid.
    
    The padded regions (outside HxW) have all 0s across the 10 channels.
    This reconstructs the exact original shape.
    """
    example = []
    _, channels, height, width = tensor.shape
    for row in range(height):
        cells = []
        for col in range(width):
            colors = [c for c in range(channels) if tensor[0, c, row, col] >= 0.5]
            cells.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        # pop trailing 10s (empty cells)
        while cells and cells[-1] == 10:
            cells.pop(-1)
        example.append(cells)
    
    # pop trailing empty rows
    while example and not example[-1]:
        example.pop(-1)
        
    return example
