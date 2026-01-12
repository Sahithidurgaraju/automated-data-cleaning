import os
import requests

# ============ CONFIG ============
OWNER = "Sahithidurgaraju"
REPO = "automated-data-cleaning"
RELEASE_TAG = "latest-images"
PLOTS_DIR = "plots"
# =================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not found")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# ---- Step 1: Get or create release ----
release_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{RELEASE_TAG}"
resp = requests.get(release_url, headers=HEADERS)

if resp.status_code == 404:
    resp = requests.post(
        f"https://api.github.com/repos/{OWNER}/{REPO}/releases",
        headers=HEADERS,
        json={
            "tag_name": RELEASE_TAG,
            "name": "Latest Dashboard Plots"
        }
    )

resp.raise_for_status()
release = resp.json()
upload_url = release["upload_url"].split("{")[0]

# ---- Step 2: Delete old assets ----
for asset in release.get("assets", []):
    requests.delete(asset["url"], headers=HEADERS)

print("Old plot images deleted")

# ---- Step 3: Upload new images ----
if not os.path.exists(PLOTS_DIR):
    print("⚠ plots directory not found")
    exit(0)

uploaded = 0

for folder in os.listdir(PLOTS_DIR):
    folder_path = os.path.join(PLOTS_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            local_path = os.path.join(folder_path, file)

            # Keep folder name in release
            upload_name = f"{folder}/{file}"

            with open(local_path, "rb") as f:
                r = requests.post(
                    upload_url,
                    headers=HEADERS,
                    params={"name": upload_name},
                    data=f.read()
                )

            if r.status_code in (200, 201):
                uploaded += 1
            else:
                print(f"Failed to upload {upload_name}")

print(f"Uploaded {uploaded} plot images successfully")
