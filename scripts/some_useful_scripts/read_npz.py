import numpy as np

def read_npz(file_path):
    """
    Read an NPZ file and return its contents.
    
    Args:
        file_path (str): Path to the NPZ file
        
    Returns:
        dict: Dictionary containing the arrays from the NPZ file
    """
    data = np.load(file_path)
    
    # Convert to dictionary if it's a NpzFile object
    if isinstance(data, np.lib.npyio.NpzFile):
        data_dict = {key: data[key] for key in data.files}
        data.close()
        return data_dict
    
    return data

# Example usage
if __name__ == "__main__":
    npz_file = "/home/yhy/IsaacLabExtensionTemplate/expert_data_episodes/0a423411-d892-4253-8b1c-392b0fbdb0b3-34.npz"
    arrays = read_npz(npz_file)
    
    # Print available arrays
    for key, value in arrays.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")