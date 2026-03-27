#!/usr/bin/env python3
"""
check_links.py — crawl burning-cost.github.io for broken internal links.

Usage:
    python scripts/check_links.py [--output broken_links.json]

Strategy:
1. Fetch sitemap.xml to get all canonical URLs.
2. For each page, parse the HTML and collect internal hrefs.
3. Resolve relative and absolute internal links.
4. HEAD-request each unique internal URL; record anything that isn't 200.
5. Report broken links grouped by referring page.

Rate limit: 0.1s between requests to avoid GitHub Pages throttling.
"""

import argparse
import json
import time
import sys
from collections import defaultdict
from urllib.parse import urljoin, urlparse, urlunparse
import urllib.request
import urllib.error
from xml.etree import ElementTree
from html.parser import HTMLParser


BASE = "https://burning-cost.github.io"
SITEMAP = f"{BASE}/sitemap.xml"
DELAY = 0.1  # seconds between requests


class LinkExtractor(HTMLParser):
    """Minimal HTML parser — pulls href from <a> tags."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


def normalise(url: str) -> str:
    """Strip fragments; ensure trailing slash on path-only URLs."""
    p = urlparse(url)
    # Drop fragment
    clean = urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))
    return clean


def is_internal(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "" or parsed.netloc == "burning-cost.github.io"


def fetch(url: str, method: str = "HEAD") -> tuple[int, str]:
    """Return (status_code, final_url). Returns (0, url) on error."""
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "BurningCostLinkChecker/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.url
    except urllib.error.HTTPError as e:
        return e.code, url
    except urllib.error.URLError as e:
        return 0, url
    except Exception as e:
        return 0, url


def fetch_html(url: str) -> tuple[int, str]:
    """GET a page; return (status, body_text)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BurningCostLinkChecker/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def get_sitemap_urls() -> list[str]:
    print(f"Fetching sitemap: {SITEMAP}", flush=True)
    status, body = fetch_html(SITEMAP)
    if status != 200:
        print(f"  ERROR: sitemap returned {status}", flush=True)
        sys.exit(1)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ElementTree.fromstring(body)
    urls = [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]
    print(f"  Found {len(urls)} URLs in sitemap", flush=True)
    return urls


def extract_internal_links(html: str, page_url: str) -> list[str]:
    parser = LinkExtractor(page_url)
    parser.feed(html)
    resolved = []
    for href in parser.links:
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absolute = urljoin(page_url, href)
        if is_internal(absolute):
            resolved.append(normalise(absolute))
    return resolved


def main():
    ap = argparse.ArgumentParser(description="Check internal links on burning-cost.github.io")
    ap.add_argument("--output", default="broken_links.json", help="JSON output file")
    ap.add_argument("--max-pages", type=int, default=0, help="Limit pages crawled (0 = all)")
    args = ap.parse_args()

    sitemap_urls = get_sitemap_urls()

    pages_to_crawl = sitemap_urls
    if args.max_pages:
        pages_to_crawl = pages_to_crawl[: args.max_pages]

    # Map: link_target -> set of referring pages
    link_sources: dict[str, set[str]] = defaultdict(set)

    # Phase 1: crawl each sitemap page and collect all internal links
    print(f"\nPhase 1: Crawling {len(pages_to_crawl)} pages to collect links...", flush=True)
    all_internal_targets: set[str] = set()

    for i, page_url in enumerate(pages_to_crawl, 1):
        if i % 25 == 0 or i == 1:
            print(f"  [{i}/{len(pages_to_crawl)}] {page_url}", flush=True)
        status, html = fetch_html(page_url)
        if status == 200:
            links = extract_internal_links(html, page_url)
            for link in links:
                link_sources[link].add(page_url)
                all_internal_targets.add(link)
        else:
            print(f"  WARN: {page_url} returned {status}", flush=True)
        time.sleep(DELAY)

    # Also add sitemap URLs themselves as targets to verify
    for url in sitemap_urls:
        all_internal_targets.add(normalise(url))

    print(f"\n  Collected {len(all_internal_targets)} unique internal URLs to verify", flush=True)

    # Phase 2: HEAD-check every unique internal URL
    print(f"\nPhase 2: Checking {len(all_internal_targets)} URLs...", flush=True)
    broken: dict[str, dict] = {}  # url -> {status, referred_by}
    ok_count = 0

    sorted_targets = sorted(all_internal_targets)
    for i, url in enumerate(sorted_targets, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(sorted_targets)}] checking...", flush=True)
        status, final_url = fetch(url)
        if status == 200:
            ok_count += 1
        else:
            referred_by = sorted(link_sources.get(url, set()))
            broken[url] = {
                "status": status,
                "referred_by": referred_by,
            }
            print(f"  BROKEN [{status}] {url}", flush=True)
            if referred_by:
                for ref in referred_by[:3]:
                    print(f"    <- {ref}", flush=True)
        time.sleep(DELAY)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"  Pages crawled:    {len(pages_to_crawl)}", flush=True)
    print(f"  Unique URLs:      {len(all_internal_targets)}", flush=True)
    print(f"  OK (200):         {ok_count}", flush=True)
    print(f"  Broken:           {len(broken)}", flush=True)
    print(f"{'='*60}", flush=True)

    if broken:
        print("\nBroken links:", flush=True)
        for url, info in sorted(broken.items()):
            print(f"  [{info['status']}] {url}", flush=True)
            for ref in info["referred_by"][:5]:
                print(f"    <- {ref}", flush=True)

    # Write JSON
    output = {
        "base": BASE,
        "pages_crawled": len(pages_to_crawl),
        "unique_urls_checked": len(all_internal_targets),
        "ok_count": ok_count,
        "broken_count": len(broken),
        "broken": broken,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}", flush=True)

    return 0 if not broken else 1


if __name__ == "__main__":
    sys.exit(main())
