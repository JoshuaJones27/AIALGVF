import kagglehub
import os
import shutil

# Download the dataset
print("📥 Downloading Kaggle dataset...")
path = kagglehub.dataset_download("wingsdong/lettuce-diseases-and-pests")
print("✅ Dataset downloaded to:", path)

# Target folder to store the dataset with original structure
target_dir = "dataset/kaggle_dataset"
os.makedirs(target_dir, exist_ok=True)

# File extensions to ignore
ignore_exts = (".mp4",)

def should_ignore(file):
    return file.lower().endswith(ignore_exts)

def copytree_with_ignore(src, dst):
    for root, dirs, files in os.walk(src):
        # Compute the relative path from the source root
        rel_path = os.path.relpath(root, src)
        # Compute the corresponding destination path
        dst_path = os.path.join(dst, rel_path) if rel_path != "." else dst
        os.makedirs(dst_path, exist_ok=True)
        for file in files:
            if not should_ignore(file):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_path, file)
                shutil.copy2(src_file, dst_file)
                print("✔️", os.path.relpath(dst_file, dst))

print("\n📂 Copying dataset (preserving folder structure, ignoring .mp4 files):")
copytree_with_ignore(path, target_dir)
print(f"\n✅ Done. Dataset copied to: {target_dir}")
