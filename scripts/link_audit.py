#!/usr/bin/env python3
"""
Broken link audit for the Burning Cost Jekyll site.

Checks:
1. Internal broken links — href/markdown links pointing to non-existent pages
2. Orphaned pages — files not linked from anywhere
3. Frontmatter validity — all _posts require layout, title, date

Run from repo root:
    python3 scripts/link_audit.py
"""

import os
import re
import sys
import yaml
from pathlib import Path
from urllib.parse import urlparse

SITE_ROOT = Path(__file__).parent.parent
POSTS_DIR = SITE_ROOT / "_posts"
PERMALINK_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)\.md$")

# Jekyll config: permalink format is /:year/:month/:day/:title/
# We replicate that to map post filenames -> expected URLs


def post_filename_to_url(filename: str) -> str:
    """Convert a _posts filename to its Jekyll URL path."""
    m = PERMALINK_PATTERN.match(filename)
    if not m:
        return None
    year, month, day, slug = m.groups()
    return f"/{year}/{month}/{day}/{slug}/"


def frontmatter_from_file(path: Path) -> tuple[dict, str]:
    """Parse YAML frontmatter from a .md or .html file. Returns (fm_dict, body)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end]
    body = text[end + 4:]
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


def collect_all_pages() -> dict[str, Path]:
    """
    Build a map of known URL paths -> source file for all Jekyll pages/posts.
    Keys are URL paths like /blog/ or /2026/03/01/some-slug/
    """
    pages = {}

    # Posts: derive URL from filename
    for f in POSTS_DIR.iterdir():
        if f.suffix == ".md":
            url = post_filename_to_url(f.name)
            if url:
                pages[url] = f
            # Also allow the file to override permalink via frontmatter
            fm, _ = frontmatter_from_file(f)
            if fm.get("permalink"):
                override = fm["permalink"]
                if not override.endswith("/"):
                    override += "/"
                pages[override] = f

    # All other .md and .html files (not in _posts, not prefixed with _)
    for f in SITE_ROOT.rglob("*.md"):
        parts = f.relative_to(SITE_ROOT).parts
        # Skip _posts (handled above), _layouts, _includes, vendor, workbench, scripts
        if parts[0].startswith("_") or parts[0] in ("vendor", "workbench", "scripts", "node_modules"):
            continue
        fm, _ = frontmatter_from_file(f)
        if fm.get("permalink"):
            perm = fm["permalink"]
            if not perm.endswith("/"):
                perm += "/"
            pages[perm] = f
        else:
            # Derive URL from path: about.md -> /about/
            rel = f.relative_to(SITE_ROOT)
            stem = rel.with_suffix("")
            if stem.name == "index":
                url = "/" + "/".join(stem.parent.parts) + "/"
                url = url.replace("//", "/")
            else:
                url = "/" + "/".join(stem.parts) + "/"
            pages[url] = f

    for f in SITE_ROOT.rglob("*.html"):
        parts = f.relative_to(SITE_ROOT).parts
        if parts[0].startswith("_") or parts[0] in ("vendor", "workbench", "scripts", "node_modules"):
            continue
        fm, _ = frontmatter_from_file(f)
        if fm.get("permalink"):
            perm = fm["permalink"]
            if not perm.endswith("/"):
                perm += "/"
            pages[perm] = f
        else:
            rel = f.relative_to(SITE_ROOT)
            stem = rel.with_suffix("")
            if stem.name == "index":
                url = "/" + "/".join(stem.parent.parts) + "/"
                url = url.replace("//", "/")
                if url == "//":
                    url = "/"
            else:
                url = "/" + "/".join(stem.parts) + "/"
            pages[url] = f

    # Also add known virtual pages that Jekyll generates
    pages["/"] = SITE_ROOT / "index.html"
    pages["/404.html"] = SITE_ROOT / "404.html"
    pages["/feed.xml"] = None  # generated
    pages["/sitemap.xml"] = None  # generated
    pages["/robots.txt"] = SITE_ROOT / "robots.txt"

    return pages


# Patterns to extract links from source files
HREF_RE = re.compile(r'href=["\']([^"\'#\s][^"\']*)["\']')
MD_LINK_RE = re.compile(r'\[(?:[^\]]*)\]\(([^)#\s][^)]*)\)')


def extract_links_from_file(path: Path) -> list[str]:
    """Extract all link targets from a file (both markdown and HTML href)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    links = []
    links += HREF_RE.findall(text)
    links += MD_LINK_RE.findall(text)
    return links


def is_internal(href: str) -> bool:
    """Return True if href is a site-internal link (not external, not Liquid template, not noise)."""
    if href.startswith(("http://", "https://", "mailto:", "tel:", "javascript:", "//")):
        return False
    # Skip Liquid template expressions - these render correctly at build time
    if "{%" in href or "{{" in href:
        return False
    # Skip single-character hrefs (false positives from code blocks, e.g. fun(m))
    if len(href.strip()) <= 1:
        return False
    return True


def normalise_internal(href: str, source_url: str) -> str:
    """
    Normalise an internal href relative to source_url.
    Returns a URL path with trailing slash (for directories/pages).
    """
    # Strip query and fragment
    href = href.split("?")[0].split("#")[0]
    if not href:
        return None
    if href.startswith("/"):
        path = href
    else:
        # Relative: resolve against source directory
        base = source_url.rsplit("/", 1)[0] + "/"
        path = base + href

    # Normalise .. and .
    parts = []
    for part in path.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part == ".":
            pass
        else:
            parts.append(part)
    path = "/".join(parts)
    if not path.startswith("/"):
        path = "/" + path

    # If it looks like a page (no extension or .html), ensure trailing slash
    _, ext = os.path.splitext(path)
    if not ext or ext == ".html":
        if not path.endswith("/"):
            path += "/"

    return path


def url_for_file(f: Path, pages: dict[str, Path]) -> str:
    """Given a source file, find its URL in the pages dict."""
    for url, pf in pages.items():
        if pf == f:
            return url
    return None


def audit_frontmatter(posts_dir: Path) -> list[dict]:
    """Check all posts for required frontmatter fields."""
    issues = []
    required = {"layout", "title", "date"}
    for f in sorted(posts_dir.iterdir()):
        if f.suffix != ".md":
            continue
        fm, _ = frontmatter_from_file(f)
        missing = required - set(fm.keys())
        if missing:
            issues.append({"file": f.name, "missing": sorted(missing)})
        # Check for empty title
        if "title" in fm and not str(fm.get("title", "")).strip():
            issues.append({"file": f.name, "missing": ["title (empty)"]})
    return issues


def main():
    print(f"Site root: {SITE_ROOT}")
    print("Collecting pages...")
    pages = collect_all_pages()
    print(f"  {len(pages)} known URL paths")

    # URL set for lookup (normalise trailing slash)
    known_urls = set()
    for url in pages:
        known_urls.add(url)
        if url.endswith("/"):
            known_urls.add(url[:-1])  # also allow without trailing slash
        else:
            known_urls.add(url + "/")

    # Also add asset paths we don't parse (images, css, js)
    # These won't trigger false positives for /assets/...
    SKIP_PREFIXES = ("/assets/", "/feed", "/sitemap", "/robots")

    # --- 1. Broken internal links ---
    print("\nScanning for broken internal links...")
    broken = []
    # Pages that contribute links
    source_files = set()

    # Collect all source md/html files (not _ dirs, not vendor)
    for f in SITE_ROOT.rglob("*.md"):
        parts = f.relative_to(SITE_ROOT).parts
        if parts[0].startswith("_") and parts[0] != "_posts":
            continue
        if parts[0] in ("vendor", "workbench", "scripts", "node_modules"):
            continue
        source_files.add(f)

    for f in SITE_ROOT.rglob("*.html"):
        parts = f.relative_to(SITE_ROOT).parts
        if parts[0].startswith("_"):
            continue
        if parts[0] in ("vendor", "workbench", "scripts", "node_modules"):
            continue
        source_files.add(f)

    # Track which URLs are referenced (for orphan check)
    referenced_urls: set[str] = set()

    for source_file in sorted(source_files):
        # Determine the URL this file produces
        source_url = url_for_file(source_file, pages)
        if source_url is None:
            # Try to derive it
            if source_file.name.endswith(".md"):
                rel = source_file.relative_to(SITE_ROOT)
                source_url = "/" + str(rel.with_suffix("")) + "/"
            else:
                source_url = "/"

        links = extract_links_from_file(source_file)
        for href in links:
            if not is_internal(href):
                continue
            norm = normalise_internal(href, source_url)
            if norm is None:
                continue
            # Skip assets, feeds, etc.
            if any(norm.startswith(p) for p in SKIP_PREFIXES):
                continue
            # Skip anchors-only
            if norm == source_url:
                continue

            referenced_urls.add(norm)

            if norm not in known_urls:
                broken.append({
                    "source": str(source_file.relative_to(SITE_ROOT)),
                    "href": href,
                    "resolved": norm,
                })

    # --- 2. Orphaned pages ---
    print("Checking for orphaned pages...")
    # Post files are the main concern — check if each post URL is referenced somewhere
    orphaned_posts = []
    for f in sorted(POSTS_DIR.iterdir()):
        if f.suffix != ".md":
            continue
        url = post_filename_to_url(f.name)
        if url is None:
            continue
        # Check frontmatter override
        fm, _ = frontmatter_from_file(f)
        if fm.get("permalink"):
            perm = fm["permalink"]
            if not perm.endswith("/"):
                perm += "/"
            url = perm

        if url not in referenced_urls and url.rstrip("/") not in referenced_urls:
            orphaned_posts.append({"url": url, "file": f.name})

    # --- 3. Frontmatter audit ---
    print("Auditing frontmatter...")
    fm_issues = audit_frontmatter(POSTS_DIR)

    # --- Report ---
    print("\n" + "=" * 60)
    print("BROKEN INTERNAL LINKS")
    print("=" * 60)
    if broken:
        for b in broken:
            print(f"  [{b['source']}]")
            print(f"    href: {b['href']}")
            print(f"    resolved to: {b['resolved']}")
    else:
        print("  None found.")

    print(f"\n  Total: {len(broken)} broken links")

    print("\n" + "=" * 60)
    print("ORPHANED BLOG POSTS (not linked from any source file)")
    print("=" * 60)
    # Note: most posts won't be explicitly linked — they appear via Liquid loops.
    # So we only flag this as informational; the archive iterates site.posts dynamically.
    # A true orphan would be a post missing from the archive because the archive
    # uses Liquid `site.posts` which picks up all posts automatically.
    # Real concern: posts with future dates or `published: false`
    print(f"  {len(orphaned_posts)} posts not explicitly linked in source files.")
    print("  (Expected — posts are listed via Liquid `site.posts` loop in the archive.)")
    # Check specifically for posts with published: false or future dates
    import datetime
    today = datetime.date.today()
    suppressed = []
    for f in sorted(POSTS_DIR.iterdir()):
        if f.suffix != ".md":
            continue
        fm, _ = frontmatter_from_file(f)
        if fm.get("published") is False:
            suppressed.append({"file": f.name, "reason": "published: false"})
        # Check date
        m = PERMALINK_PATTERN.match(f.name)
        if m:
            y, mo, d, _ = m.groups()
            post_date = datetime.date(int(y), int(mo), int(d))
            if post_date > today:
                suppressed.append({"file": f.name, "reason": f"future date ({post_date})"})
    if suppressed:
        print("\n  SUPPRESSED (will not appear on live site):")
        for s in suppressed:
            print(f"    {s['file']} — {s['reason']}")
    else:
        print("  No suppressed posts found.")

    print("\n" + "=" * 60)
    print("FRONTMATTER ISSUES (missing required fields)")
    print("=" * 60)
    if fm_issues:
        for issue in fm_issues:
            print(f"  {issue['file']} — missing: {', '.join(issue['missing'])}")
    else:
        print("  All posts have valid frontmatter (layout, title, date).")
    print(f"\n  Checked {len(list(POSTS_DIR.glob('*.md')))} posts")

    # Return exit code
    total_issues = len(broken) + len(fm_issues) + len([s for s in suppressed if "published: false" not in s["reason"]])
    return broken, orphaned_posts, fm_issues, suppressed


if __name__ == "__main__":
    broken, orphaned, fm_issues, suppressed = main()
    future_posts = [s for s in suppressed if "future" in s["reason"]]
    if broken or fm_issues:
        print("\nACTION REQUIRED: Fix broken links and/or frontmatter issues above.")
        sys.exit(1)
    else:
        if future_posts:
            print(f"\nINFO: {len(future_posts)} future-dated post(s) will publish when their date arrives.")
        print("\nAudit passed.")
        sys.exit(0)
