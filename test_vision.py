#!/usr/bin/env python3
"""
Quick end-to-end test for the vision pipeline.
Usage:
    GEMINI_API_KEY=AIza... python3 test_vision.py [image_path]
"""
import asyncio
import os
import sys


async def test(image_path: str):
    from services.vision_service import vision_service

    print(f"\n{'='*60}")
    print(f"Testing vision pipeline")
    print(f"Image:  {image_path}")
    print(f"Key:    {os.getenv('GEMINI_API_KEY', '')[:12]}... (masked)")
    print(f"{'='*60}\n")

    with open(image_path, "rb") as f:
        data = f.read()

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "webp": "image/webp",
    }.get(ext, "image/jpeg")

    print(f"Loaded {len(data)//1024} KB ({mime})")
    print("Running vision analysis...\n")

    result = await vision_service.analyze_image(data, mime)

    print("── Result ─────────────────────────────────────────────")
    for k, v in result.items():
        val = str(v)
        if len(val) > 200:
            val = val[:200] + "..."
        print(f"  {k:20s}: {val}")

    # Simulate what _enrich_image_background builds
    extracted    = (result.get("extracted_text") or "").strip()
    recall_terms = (result.get("recall_terms")   or "").strip()
    description  = (result.get("description")    or "").strip()
    doc_label    = result.get("document_type", "other").replace("_", " ")

    kw = []
    kw.append(doc_label)
    if extracted:
        kw.append(extracted[:200])
    if recall_terms:
        kw.append(recall_terms)
    content = "\n".join(kw)

    print(f"\n── content (keyword-search blob, first 300 chars) ─────")
    print(content[:300])

    print(f"\n── embed_text (semantic vector input) ─────────────────")
    em = []
    if description:
        em.append(description)
    if recall_terms:
        em.append(recall_terms)
    print(" ".join(em)[:200])
    print(f"\n{'='*60}\nVision pipeline OK\n")


if __name__ == "__main__":
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    img = sys.argv[1] if len(sys.argv) > 1 else None
    if not img:
        # Try images in the ExtendedMinds root
        candidates = [
            "../ashu.JPG",
            "../ios_light_rd_na@3x.png",
            "../em_pro_1024.png",
        ]
        for c in candidates:
            path = os.path.join(os.path.dirname(__file__), c)
            if os.path.exists(path):
                img = path
                break

    if not img or not os.path.exists(img):
        print(f"No image found. Pass a path: python3 test_vision.py /path/to/image.jpg")
        sys.exit(1)

    asyncio.run(test(img))
