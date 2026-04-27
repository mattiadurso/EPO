# Freeze/Unfreeze Implementation Explanation

## Overview

The `freeze_all()`, `unfreeze()`, `unfreeze_all()`, and `parameters_for()` methods are used to selectively enable/disable gradient computation for different parameters during optimization. This is crucial for multi-phase optimization strategies.

## Why Freeze/Unfreeze?

In this pose optimization system, you often want to optimize **only specific parameters** while keeping others fixed:

1. **Pose Refinement Phase**: Optimize only the new frame's pose while keeping all other poses and intrinsics frozen
2. **Global Optimization Phase**: Optimize all poses and intrinsics together
3. **Intrinsic Refinement**: Optimize camera intrinsics while keeping poses frozen

## Implementation Details

### PoseModule Methods

#### 1. `freeze_all()`
```python
def freeze_all(self):
    """Freeze all pose parameters by setting requires_grad=False."""
    self.q_param.requires_grad = False
    self.t_param.requires_grad = False
    self.t_offset.requires_grad = False
    
    if self.use_mlp:
        for param in self.mlp.parameters():
            param.requires_grad = False
```

**What it does:**
- Sets `requires_grad=False` for all rotation quaternions (`q_param`)
- Sets `requires_grad=False` for all translations (`t_param`)
- Sets `requires_grad=False` for translation offsets (`t_offset`)
- Also freezes MLP parameters if MLP-based pose refinement is enabled

**When to use:**
- Before optimizing only intrinsics
- Before local pose refinement to prevent updating global poses
- In the `add_frame()` method: `self.poses.freeze_all()`

#### 2. `unfreeze(image_name)`
```python
def unfreeze(self, image_name: str):
    """Unfreeze (enable gradients for) parameters of a specific image."""
    if image_name not in self.image_to_tensor_idx:
        raise ValueError(f"Image '{image_name}' not found in pose module.")
    
    # Enable gradients for all parameters
    self.q_param.requires_grad = True
    self.t_param.requires_grad = True
    self.t_offset.requires_grad = True
    
    if self.use_mlp:
        for param in self.mlp.parameters():
            param.requires_grad = True
```

**What it does:**
- Takes an image name as input
- Enables gradients for all parameters

**Important Note:** 
Since PyTorch parameters are shared tensors, we can't selectively freeze just one image's quaternion while keeping others frozen. Instead, we:
1. Freeze ALL parameters
2. Enable gradients for ALL parameters
3. Use `parameters_for(image_name)` to create an optimizer that only updates that specific image's parameters

**When to use:**
- After `freeze_all()` to enable selective optimization
- In the `add_frame()` method: `self.poses.unfreeze(image_name)` after `freeze_all()`

#### 3. `unfreeze_all()`
```python
def unfreeze_all(self):
    """Unfreeze all pose parameters by setting requires_grad=True."""
    self.q_param.requires_grad = True
    self.t_param.requires_grad = True
    self.t_offset.requires_grad = True
    
    if self.use_mlp:
        for param in self.mlp.parameters():
            param.requires_grad = True
```

**What it does:**
- Enables gradients for all parameters
- Returns to global optimization mode

**When to use:**
- After local optimization phase completes
- In the `add_frame()` method: `self.poses.unfreeze_all()` after pose refinement
- Before returning to full-scene optimization

#### 4. `parameters_for(image_name)`
```python
def parameters_for(self, image_name: str, recurse: bool = True):
    """Get parameters for a specific image only."""
    if image_name not in self.image_to_tensor_idx:
        raise ValueError(f"Image '{image_name}' not found in pose module.")
    
    idx = self.image_to_tensor_idx[image_name]
    params = []
    
    # Return only the parameters for this specific image index
    if self.q_param.requires_grad:
        q_single = self.q_param[idx:idx+1]
        params.append(q_single)
    
    if self.t_param.requires_grad:
        t_single = self.t_param[idx:idx+1]
        params.append(t_single)
    
    if self.t_offset.requires_grad:
        t_offset_single = self.t_offset[idx:idx+1]
        params.append(t_offset_single)
    
    if self.use_mlp and any(p.requires_grad for p in self.mlp.parameters()):
        params.extend([p for p in self.mlp.parameters() if p.requires_grad])
    
    return params
```

**What it does:**
- Extracts parameter slices for a specific image (by creating views)
- Returns only the parameters that require gradients
- These can be passed directly to an optimizer

**When to use:**
- When creating a targeted optimizer for a single image
- In the `add_frame()` method: `torch.optim.Adam(self.poses.parameters_for(image_name), lr=...)`

### CameraModule Methods

#### 1. `freeze_all()`
```python
def freeze_all(self):
    """Freeze all intrinsic parameters by setting requires_grad=False."""
    self.params.requires_grad = False
```

#### 2. `unfreeze_all()`
```python
def unfreeze_all(self):
    """Unfreeze all intrinsic parameters by setting requires_grad=True."""
    self.params.requires_grad = True
```

**When to use:**
- `freeze_all()`: Before optimizing only poses
- `unfreeze_all()`: Before optimizing intrinsics again

## Usage Example from `add_frame()`

```python
# Step 1: Freeze all poses and intrinsics
self.poses.freeze_all()
self.poses.unfreeze(image_name)      # Re-enable only new frame's pose
self.intrinsics.freeze_all()          # Keep all intrinsics fixed

# Step 2: Create optimizer for only the new frame's pose
pose_optimizer = torch.optim.Adam(
    self.poses.parameters_for(image_name),
    lr=self.t_lr,
)

# Step 3: Optimize only the new frame's pose
for _ in range(num_pose_iters):
    pose_optimizer.zero_grad()
    # ... forward pass and loss computation ...
    loss.backward()
    pose_optimizer.step()

# Step 4: Restore normal optimization
self.poses.unfreeze_all()
self.intrinsics.unfreeze_all()
```

## Technical Notes

### Parameter Sharing Limitation
In PyTorch, when parameters are stored as shared tensors (like `self.q_param` which contains all quaternions), you cannot freeze individual elements while keeping others unfrozen. This is because they all belong to the same tensor object.

**Solution:** 
- Use `freeze_all()` / `unfreeze()` on the entire parameter tensor
- Use `parameters_for(image_name)` to create parameter slices for the optimizer
- The optimizer will only update the parameters explicitly passed to it

### Gradient Flow
- When a parameter has `requires_grad=False`, gradients are not computed through it during backpropagation
- When `requires_grad=True`, `.backward()` will compute gradients for it
- The optimizer only updates parameters that have `requires_grad=True` AND are included in its param_groups

## Summary

| Method | Purpose | Called On |
|--------|---------|-----------|
| `freeze_all()` | Disable all gradients | `poses`, `intrinsics` |
| `unfreeze()` | Enable gradient for specific image | `poses` only |
| `unfreeze_all()` | Enable all gradients | `poses`, `intrinsics` |
| `parameters_for()` | Get params for one image | `poses` only |
