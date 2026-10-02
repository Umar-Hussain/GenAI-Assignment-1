import torch
import torchvision.transforms.functional as F
import random

def apply_salt_pepper(img_tensor, prob):
    noisy = img_tensor.clone()
    noise = torch.rand(noisy.shape[1:])
    salt = noise < (prob / 2)
    pepper = (noise > (prob / 2)) & (noise < prob)
    noisy[:, salt] = 1.0
    noisy[:, pepper] = 0.0
    return noisy, {"type": "salt_pepper", "prob": prob}

def apply_gaussian_blur(img_tensor, kernel_size, sigma):
    blurred = F.gaussian_blur(img_tensor, kernel_size, [sigma, sigma])
    return blurred, {"type": "gaussian_blur", "kernel_size": kernel_size, "sigma": sigma}

def apply_rectangular_occlusion(img_tensor, num_rects, coverage, rng=None):
    occluded = img_tensor.clone()
    _, h, w = occluded.shape
    total_area = h * w
    per_rect_area = int(total_area * coverage / num_rects)
    coords = []

    for _ in range(num_rects):
        rw = random.randint(max(1, int(per_rect_area ** 0.5) // 2), min(w, int(per_rect_area ** 0.5) * 2))
        rh = max(1, per_rect_area // rw)
        rw, rh = min(rw, w), min(rh, h)
        x = random.randint(0, w - rw)
        y = random.randint(0, h - rh)
        occluded[:, y:y+rh, x:x+rw] = 0.0
        coords.append([x, y, rw, rh])

    return occluded, {"type": "occlusion", "num_rects": num_rects, "coverage": coverage, "coords": coords}
