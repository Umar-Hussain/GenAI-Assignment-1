import json
import random

def generate_manifest(dataset_size, split, seed=42, save_path=None):
    random.seed(seed)
    manifest = []
    
    severities = {
        "salt_pepper": [{"prob": 0.03}, {"prob": 0.08}, {"prob": 0.15}],
        "gaussian_blur": [{"kernel_size": 3, "sigma": 0.7}, {"kernel_size": 5, "sigma": 1.5}, {"kernel_size": 7, "sigma": 2.5}],
        "occlusion": [{"num_rects": 1, "coverage": 0.10}, {"num_rects": 2, "coverage": 0.20}, {"num_rects": 3, "coverage": 0.35}]
    }
    
    types = ["clean", "salt_pepper", "gaussian_blur", "occlusion"]
    
    for i in range(dataset_size):
        c_type = random.choice(types)
        params = {"type": c_type}
        
        if c_type != "clean":
            if split == 'test':
                sev = random.choice(severities[c_type])
                params.update(sev)
            else:
                if c_type == "salt_pepper":
                    params["prob"] = random.uniform(0.02, 0.15)
                elif c_type == "gaussian_blur":
                    params["kernel_size"] = random.choice([3, 5, 7])
                    params["sigma"] = random.uniform(0.5, 2.5)
                elif c_type == "occlusion":
                    params["num_rects"] = random.randint(1, 3)
                    params["coverage"] = random.uniform(0.10, 0.35)
                    
        manifest.append(params)
        
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
    return manifest
