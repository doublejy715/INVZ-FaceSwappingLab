import json

# load json file
def load_info_file(path):
    with open(path, 'r+') as f:
        info = json.load(f)
    return info

def save_info_file(path, add_info):
    with open(path, 'w') as f:
        json.dump(add_info, f, indent=4)