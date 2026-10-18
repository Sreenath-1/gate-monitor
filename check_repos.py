import urllib.request
import json
import sys

repos = [
    "agungfathur/Facemask-Detection-MTCNN-ONNX",
    "ZJW-92/ML_face_mask_detection",
    "waittim/mask-detector",
    "ShengG/FaceMaskDetection",
    "AIZOOTech/FaceMaskDetection"
]

def check_repo(repo):
    print(f"\nChecking repo: {repo}")
    try:
        url = f"https://api.github.com/repos/{repo}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            lic = data.get('license')
            print(f"License: {lic.get('name') if lic else 'None'}")
            default_branch = data.get('default_branch', 'main')
            
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{default_branch}?recursive=1"
        req = urllib.request.Request(tree_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            tree_data = json.loads(response.read().decode())
            for item in tree_data.get('tree', []):
                if item['path'].endswith('.onnx'):
                    print(f"Found ONNX: {item['path']} (Size: {item.get('size', 'Unknown')} bytes)")
    except Exception as e:
        print(f"Error checking repo: {e}")

for repo in repos:
    check_repo(repo)
