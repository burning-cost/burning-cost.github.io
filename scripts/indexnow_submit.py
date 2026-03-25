#!/usr/bin/env python3
"""
IndexNow URL submission script for burning-cost.github.io.

Reads the live sitemap, extracts all URLs, and submits them to the IndexNow
API endpoint (api.indexnow.org). This covers Bing, Yandex, and other IndexNow
participants. Google does NOT support IndexNow — use Google Search Console for
that separately.

Usage:
    python3 scripts/indexnow_submit.py
    python3 scripts/indexnow_submit.py --dry-run   # Print URLs without submitting
    python3 scripts/indexnow_submit.py --batch-size 100  # Override batch size

Re-run this whenever you publish new content to get fast indexation.

Requirements: Python 3.6+ stdlib only (urllib, json, xml).
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

# Site configuration
HOST = "burning-cost.github.io"
KEY = "a6d4d5d9c88365753c1b47447d878855"
KEY_LOCATION = f"https://{HOST}/{KEY}.txt"
SITEMAP_URL = f"https://{HOST}/sitemap.xml"
INDEXNOW_ENDPOINT = "https://api.indexnow.org/indexnow"

# IndexNow max per batch is 10,000 but keep smaller for reliability
DEFAULT_BATCH_SIZE = 500


def fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    """Fetch sitemap XML and extract all <loc> URLs."""
    print(f"Fetching sitemap: {sitemap_url}")
    req = urllib.request.Request(
        sitemap_url,
        headers={"User-Agent": "IndexNow-Submit/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)

    # Sitemap XML uses a namespace — handle both namespaced and bare tags
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text.strip() for loc in root.findall("sm:url/sm:loc", ns)]

    if not urls:
        # Fallback: try without namespace
        urls = [loc.text.strip() for loc in root.findall(".//loc")]

    print(f"Found {len(urls)} URLs in sitemap")
    return urls


def submit_batch(urls: list[str], batch_num: int, total_batches: int, dry_run: bool) -> bool:
    """Submit a single batch to IndexNow. Returns True on success."""
    payload = {
        "host": HOST,
        "key": KEY,
        "keyLocation": KEY_LOCATION,
        "urlList": urls,
    }

    print(f"\nBatch {batch_num}/{total_batches}: {len(urls)} URLs")
    for url in urls:
        print(f"  {url}")

    if dry_run:
        print("  [dry-run] Skipping submission")
        return True

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        INDEXNOW_ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "IndexNow-Submit/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            response_body = resp.read().decode("utf-8", errors="replace")
        print(f"  Response: HTTP {status} — {response_body or '(empty body)'}")
        # 200 = OK, 202 = Accepted (async processing)
        return status in (200, 202)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"  ERROR: HTTP {exc.code} — {error_body}", file=sys.stderr)
        return False
    except urllib.error.URLError as exc:
        print(f"  ERROR: {exc.reason}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit sitemap URLs to IndexNow")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs without submitting to IndexNow",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"URLs per batch (default: {DEFAULT_BATCH_SIZE}, max: 10000)",
    )
    args = parser.parse_args()

    if args.batch_size > 10_000:
        print("ERROR: IndexNow batch size cannot exceed 10,000", file=sys.stderr)
        return 1

    try:
        urls = fetch_sitemap_urls(SITEMAP_URL)
    except Exception as exc:
        print(f"ERROR fetching sitemap: {exc}", file=sys.stderr)
        return 1

    if not urls:
        print("No URLs found in sitemap — nothing to submit.")
        return 0

    # Split into batches
    batches = [urls[i : i + args.batch_size] for i in range(0, len(urls), args.batch_size)]
    total_batches = len(batches)

    print(f"\nSubmitting {len(urls)} URLs in {total_batches} batch(es) to IndexNow")
    print(f"Endpoint: {INDEXNOW_ENDPOINT}")
    print(f"Key location: {KEY_LOCATION}")
    if args.dry_run:
        print("Mode: DRY RUN")

    failed = 0
    for i, batch in enumerate(batches, start=1):
        ok = submit_batch(batch, i, total_batches, args.dry_run)
        if not ok:
            failed += 1

    print(f"\nDone. {total_batches - failed}/{total_batches} batches succeeded.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
