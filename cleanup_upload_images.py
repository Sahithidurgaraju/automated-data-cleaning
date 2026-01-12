import os
import requests
from urllib.parse import quote
# ============ CONFIG ============
OWNER = "Sahithidurgaraju"
REPO = "automated-data-cleaning"
RELEASE_TAG = "latest-images"
PLOTS_DIR = "plots"
DASHBOARD_DIR = "dashboard_reports"
# =================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not found")

# ---------- Headers ----------
API_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

UPLOAD_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Content-Type": "application/octet-stream"
}

# ---------- Step 1: Get or create release ----------
release_url = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{RELEASE_TAG}"
resp = requests.get(release_url, headers=API_HEADERS)

if resp.status_code == 404:
    print("Release not found. Creating release...")
    resp = requests.post(
        f"https://api.github.com/repos/{OWNER}/{REPO}/releases",
        headers=API_HEADERS,
        json={
            "tag_name": RELEASE_TAG,
            "name": "Latest Dashboard Images",
            "draft": False,
            "prerelease": False
        }
    )

resp.raise_for_status()
release = resp.json()

upload_url = release["upload_url"].split("{")[0]
print(f"Using upload URL: {upload_url}")

# ---------- Step 2: Delete old assets ----------
assets = release.get("assets", [])
for asset in assets:
    del_resp = requests.delete(asset["url"], headers=API_HEADERS)
    if del_resp.status_code == 204:
        print(f"Deleted old asset: {asset['name']}")
    else:
        print(f"Failed to delete asset: {asset['name']}")


# ---------- Step 3: Upload new images ----------
if not os.path.exists(PLOTS_DIR):
    print("⚠ plots directory not found. Nothing to upload.")
    exit(0)

uploaded = 0

for folder in os.listdir(PLOTS_DIR):
    folder_path = os.path.join(PLOTS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith((".png")):
            local_path = os.path.join(folder_path, file)

            # sanitize filename
            safe_folder = folder.replace(" ", "_")
            safe_file = file.replace(" ", "_")
            asset_name = f"{safe_folder}__{safe_file}"

            # URL encode name
            encoded_name = quote(asset_name)
            final_upload_url = f"{upload_url}?name={encoded_name}"

            with open(local_path, "rb") as f:
                r = requests.post(
                    final_upload_url,
                    headers=UPLOAD_HEADERS,
                    data=f.read()
                )

            if r.status_code in (200, 201):
                print(f"Uploaded: {asset_name}")
                uploaded += 1
            else:
                print(
                    f"Failed: {asset_name} "
                    f"→ Status {r.status_code} | {r.text}"
                )

print(f"Uploaded {uploaded} plot images successfully")

if not os.path.exists(DASHBOARD_DIR):
    print("⚠ dashboard_reports directory not found. Nothing to upload.")
    exit(0)

uploaded = 0

for file in os.listdir(DASHBOARD_DIR):
    if file.lower().endswith(".pdf"):
        local_path = os.path.join(DASHBOARD_DIR, file)

        safe_name = file.replace(" ", "_")
        encoded_name = quote(safe_name)
        final_upload_url = f"{upload_url}?name={encoded_name}"

        with open(local_path, "rb") as f:
            r = requests.post(
                final_upload_url,
                headers=UPLOAD_HEADERS,
                data=f.read()
            )

        if r.status_code in (200, 201):
            print(f"Uploaded: {safe_name}")
            uploaded += 1

        else:
            print(
                f" Failed: {safe_name} "
                f"→ Status {r.status_code} | {r.text}"
            )

print(f"Uploaded {uploaded} PDFs")
