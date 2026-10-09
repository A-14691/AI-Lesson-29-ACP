import base64, os
from datetime import datetime
import requests
from config import HF_API_KEY

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
MODELS = [
    "google/gemma-3-27b-it",
]
VALID = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp", ".webp": "image/webp", ".gif": "image/gif"}


def data_url(data, ext):
    return f"data:{MIME.get(ext.lower(), 'image/jpeg')};base64," + base64.b64encode(data).decode("utf-8")


def get_images(path):
    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in VALID:
        return [path]
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in VALID]
    raise FileNotFoundError(f"Folder or image file not found: {path}")


def caption_image(path):
    with open(path, "rb") as f:
        image = f.read()
    ext = os.path.splitext(path)[1].lower()
    last_error = "Unknown API error"
    for model in MODELS:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {"type": "image_url", "image_url": {"url": data_url(image, ext)}},
                ],
            }],
            "max_tokens": 60,
            "temperature": 0.2,
        }
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
            if r.status_code != 200:
                try:
                    error_json = r.json()
                    last_error = error_json.get("error", {}).get("message") or str(error_json)
                except Exception:
                    last_error = r.text[:200] or f"HTTP {r.status_code}"
                continue
            caption = (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            if caption:
                return caption
            last_error = "No caption returned"
        except Exception as e:
            last_error = str(e)
    return f"Caption failed: {last_error}"


def main():
    path = input("📁 Enter image folder or file path (default: images): ").strip() or "images"
    try:
        images = get_images(path)
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        return
    if not images:
        print(f"\n⚠️ No valid images found in {path}\n")
        return
    results = []
    for img in images:
        caption = caption_image(img)
        results.append((os.path.basename(img), caption))
        print(f"{os.path.basename(img)}: {caption}")
    summary = os.path.join(os.path.dirname(path) if os.path.isfile(path) else path, "captions_summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"AI Image Caption Summary\nFolder: {os.path.dirname(path) if os.path.isfile(path) else path}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for name, caption in results:
            f.write(f"{name}: {caption}\n")
    print(f"\n✅ Summary saved to {summary}\n")


if __name__ == "__main__":
    main()
