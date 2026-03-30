import numpy as np
import mlx.core as mx

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def make_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> mx.array:
    return mx.array(np.linspace(beta_start, beta_end, num_steps, dtype=np.float32))

def patchify_frames(frames: mx.array, patch_size: int) -> mx.array:
    b, t, h, w, c = frames.shape
    ph = h // patch_size
    pw = w // patch_size
    x = frames.reshape(b, t, ph, patch_size, pw, patch_size, c)
    x = mx.transpose(x, (0,1,2,4,3,5,6))
    return x.reshape(b, t, ph*pw, patch_size*patch_size*c)

def unpatchify_tokens(tokens: mx.array, patch_size: int, image_size: int, channels: int) -> mx.array:
    b, n, _ = tokens.shape
    ph = image_size // patch_size
    pw = image_size // patch_size
    x = tokens.reshape(b, ph, pw, patch_size, patch_size, channels)
    x = mx.transpose(x, (0,1,3,2,4,5))
    return x.reshape(b, image_size, image_size, channels)

def _draw_square(canvas, cx, cy, size, value):
    h, w = canvas.shape[:2]
    x0 = max(0, cx-size); x1 = min(w, cx+size)
    y0 = max(0, cy-size); y1 = min(h, cy+size)
    canvas[y0:y1, x0:x1] = value

def _draw_circle(canvas, cx, cy, radius, value):
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    canvas[(xx-cx)**2 + (yy-cy)**2 <= radius**2] = value

def sample_moving_shapes_batch(batch_size, seq_len, image_size, channels=1):
    frames = np.zeros((batch_size, seq_len, image_size, image_size, channels), dtype=np.float32)
    actions = np.zeros((batch_size, seq_len-1, 4), dtype=np.float32)
    for b in range(batch_size):
        x = np.random.randint(6, image_size-6)
        y = np.random.randint(6, image_size-6)
        vx = np.random.choice([-2,-1,1,2])
        vy = np.random.choice([-2,-1,1,2])
        shape_type = np.random.choice([0,1])
        for t in range(seq_len):
            canvas = np.zeros((image_size, image_size), dtype=np.float32)
            if shape_type == 0:
                _draw_square(canvas, x, y, 3, 1.0)
            else:
                _draw_circle(canvas, x, y, 3, 1.0)
            frames[b, t, :, :, 0] = canvas
            if t < seq_len - 1:
                actions[b, t] = np.array([vx, vy, float(shape_type), 1.0], dtype=np.float32)
            x += vx; y += vy
            if x < 4 or x > image_size-4:
                vx *= -1; x = np.clip(x, 4, image_size-4)
            if y < 4 or y > image_size-4:
                vy *= -1; y = np.clip(y, 4, image_size-4)
    return mx.array(frames), mx.array(actions)

def batch_iterator(num_batches, batch_size, seq_len, image_size, channels=1):
    for _ in range(num_batches):
        frames, actions = sample_moving_shapes_batch(batch_size, seq_len, image_size, channels)
        yield {"frames": frames, "actions": actions}
