import os

# Paths
dataset_root = "running_detection.v1i.yolov8"
splits = ["train", "valid", "test"]

# Mapping: old -> new
mapping = {
    0: 0,  # normal
    1: 0,
    2: 1,  # running
    3: 0,
    4: 0,
    5: 1,
    6: 1
}


for split in splits:
    labels_dir = os.path.join(dataset_root, split, "labels")
    for file_name in os.listdir(labels_dir):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(labels_dir, file_name)
        
        new_lines = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                old_class = int(parts[0])
                if old_class in mapping:
                    new_class = mapping[old_class]
                    parts[0] = str(new_class)
                    new_lines.append(" ".join(parts))
        
        # Overwrite file with new labels
        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))
