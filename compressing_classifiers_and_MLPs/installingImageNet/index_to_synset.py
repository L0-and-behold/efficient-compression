import json
import os
from pathlib import Path

def load_imagenet_mapping(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

def index_to_synset(index, mapping):
    index_str = str(index)
    if index_str in mapping:
        return mapping[index_str][0]  # First element is synset ID
    else:
        raise ValueError(f"Index {index} not found in mapping")

def index_to_name(index, mapping):
    index_str = str(index)
    if index_str in mapping:
        return mapping[index_str][1]  # Second element is human name
    else:
        raise ValueError(f"Index {index} not found in mapping")

def create_index_to_synset_dict(mapping):
    return {int(idx): data[0] for idx, data in mapping.items()}

def create_index_to_name_dict(mapping):
    return {int(idx): data[1] for idx, data in mapping.items()}

# Example usage
if __name__ == "__main__":

    script_dir = Path(__file__).parent
    json_file = script_dir / "imagenet_class_index.json"
    
    mapping = load_imagenet_mapping(json_file)
    
    # Method 1: Direct lookup functions
    print("Method 1 - Direct lookup:")
    print(f"Index 0 -> Synset: {index_to_synset(0, mapping)}")
    print(f"Index 0 -> Name: {index_to_name(0, mapping)}")
    print(f"Index 479 -> Synset: {index_to_synset(479, mapping)}")  # car wheel
    print(f"Index 479 -> Name: {index_to_name(479, mapping)}")
    
    # Method 2: Create lookup dicts for faster repeated access
    print("\nMethod 2 - Pre-built lookup dicts:")
    index_to_synset_dict = create_index_to_synset_dict(mapping)
    index_to_name_dict = create_index_to_name_dict(mapping)
    
    # Now you can do fast lookups
    print(f"Index 0 -> Synset: {index_to_synset_dict[0]}")
    print(f"Index 479 -> Synset: {index_to_synset_dict[479]}")
    
    # Test with multiple indices
    test_indices = [0, 1, 2, 479, 999]
    print(f"\nTest lookups:")
    for idx in test_indices:
        synset = index_to_synset_dict[idx]
        name = index_to_name_dict[idx]
        print(f"Index {idx:3d} -> {synset} -> {name}")