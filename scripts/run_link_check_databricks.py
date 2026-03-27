"""
Submit the link-check script to Databricks as a one-shot job using the SDK.
Runs on a small cluster; results are printed to job logs.
"""
import os
import sys
import json
import time
import base64

# Load Databricks credentials
env_file = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

NOTEBOOK_SOURCE = r'''
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
DELAY = 0.1

class LinkExtractor(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)


def normalise(url):
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))


def is_internal(url):
    parsed = urlparse(url)
    return parsed.netloc == "" or parsed.netloc == "burning-cost.github.io"


def fetch(url, method="HEAD"):
    req = urllib.request.Request(
        url, method=method,
        headers={"User-Agent": "BurningCostLinkChecker/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.url
    except urllib.error.HTTPError as e:
        return e.code, url
    except Exception:
        return 0, url


def fetch_html(url):
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


def get_sitemap_urls():
    print(f"Fetching sitemap: {SITEMAP}", flush=True)
    status, body = fetch_html(SITEMAP)
    if status != 200:
        print(f"ERROR: sitemap returned {status}", flush=True)
        sys.exit(1)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ElementTree.fromstring(body)
    urls = [loc.text.strip() for loc in root.findall(".//sm:loc", ns) if loc.text]
    print(f"Found {len(urls)} URLs in sitemap", flush=True)
    return urls


def extract_internal_links(html, page_url):
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


sitemap_urls = get_sitemap_urls()
pages_to_crawl = sitemap_urls
link_sources = defaultdict(set)
all_internal_targets = set()

print(f"\nPhase 1: Crawling {len(pages_to_crawl)} pages...", flush=True)
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

for url in sitemap_urls:
    all_internal_targets.add(normalise(url))

print(f"\nCollected {len(all_internal_targets)} unique internal URLs", flush=True)
print(f"\nPhase 2: Checking {len(all_internal_targets)} URLs...", flush=True)

broken = {}
ok_count = 0

for i, url in enumerate(sorted(all_internal_targets), 1):
    if i % 50 == 0 or i == 1:
        print(f"  [{i}/{len(all_internal_targets)}] checking...", flush=True)
    status, final_url = fetch(url)
    if status == 200:
        ok_count += 1
    else:
        referred_by = sorted(link_sources.get(url, set()))
        broken[url] = {"status": status, "referred_by": referred_by}
        print(f"  BROKEN [{status}] {url}", flush=True)
        for ref in referred_by[:3]:
            print(f"    <- {ref}", flush=True)
    time.sleep(DELAY)

print(f"\n{'='*60}", flush=True)
print(f"SUMMARY", flush=True)
print(f"  Pages crawled:  {len(pages_to_crawl)}", flush=True)
print(f"  Unique URLs:    {len(all_internal_targets)}", flush=True)
print(f"  OK (200):       {ok_count}", flush=True)
print(f"  Broken:         {len(broken)}", flush=True)
print(f"{'='*60}", flush=True)

result = {
    "pages_crawled": len(pages_to_crawl),
    "unique_urls_checked": len(all_internal_targets),
    "ok_count": ok_count,
    "broken_count": len(broken),
    "broken": broken,
}

print("\n===JSON_RESULT_START===")
print(json.dumps(result, indent=2))
print("===JSON_RESULT_END===")
'''

NB_PATH = "/Workspace/burning-cost/check_links_nb"

# Build notebook JSON
nb_content = {
    "nbformat": 4,
    "nbformat_minor": 2,
    "metadata": {"language_info": {"name": "python"}},
    "cells": [
        {
            "cell_type": "code",
            "source": NOTEBOOK_SOURCE,
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        }
    ],
}

print(f"Uploading notebook to {NB_PATH}...", flush=True)
nb_encoded = base64.b64encode(json.dumps(nb_content).encode()).decode()
w.workspace.import_(
    path=NB_PATH,
    content=nb_encoded,
    format=ImportFormat.JUPYTER,
    language=Language.PYTHON,
    overwrite=True,
)
print("Upload complete.", flush=True)

# Submit as a one-time run
print("Submitting job run...", flush=True)
run = w.jobs.submit(
    run_name="burning-cost-link-check",
    tasks=[
        jobs.SubmitTask(
            task_key="link_check",
            notebook_task=jobs.NotebookTask(
                notebook_path=NB_PATH,
            ),
        )
    ],
)

run_id = run.run_id
host = os.environ["DATABRICKS_HOST"].rstrip("/")
print(f"Submitted run ID: {run_id}", flush=True)
print(f"Monitor at: {host}/#job/runs/{run_id}", flush=True)

# Poll for completion
print("Waiting for completion...", flush=True)
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state.value
    result_state = run_state.state.result_state.value if run_state.state.result_state else "N/A"
    print(f"  State: {life_cycle} / {result_state}", flush=True)
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(30)

print(f"\nFinal state: {life_cycle} / {result_state}", flush=True)

# Fetch task output/logs
if run_state.tasks:
    for task in run_state.tasks:
        try:
            task_output = w.jobs.get_run_output(run_id=task.run_id)
            if task_output.notebook_output and task_output.notebook_output.result:
                print("\n=== NOTEBOOK RESULT ===")
                print(task_output.notebook_output.result)
            if task_output.logs:
                print("\n=== TASK LOGS ===")
                print(task_output.logs)
            if task_output.error:
                print(f"\n=== ERROR ===\n{task_output.error}")
            if task_output.error_trace:
                print(f"\n=== TRACE ===\n{task_output.error_trace}")
        except Exception as e:
            print(f"Could not fetch output for task {task.task_key}: {e}")
