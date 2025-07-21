"""make_episode.py – Multi‑source AI/ML podcast generator
====================================================
• Builds a resilient paper pool from 5 sources (arXiv, PapersWithCode, Semantic Scholar, OpenAlex, OpenReview)
• Picks one weighted‑random popular paper every run
• Summarises to ~1300 words with OpenAI and adds intro/outro
• Converts to 10‑min audio using Amazon Polly Neural (Matthew, conversational)
• Uploads MP3 to S3 and rewrites feed.xml (feedparser + feedgen)
• No queue/state files; every run publishes a new episode

Dependencies
------------
openai, boto3, requests, feedgen>=1.0.0, feedparser>=6.0.10,
arxiv==2.1.0, python-dateutil, pydub>=0.25.1, ffmpeg in PATH.

Environment vars expected via GitHub Actions secrets/env:
  OPENAI_API_KEY   AWS_ACCESS_KEY_ID   AWS_SECRET_ACCESS_KEY
  S3_BUCKET        AWS_REGION
Optional knobs (env vars):
  ARXIV_LOOKBACK_DAYS (default 365)
  ARXIV_MAX_RESULTS  (default 50)
"""
from __future__ import annotations

import datetime, os, random, time, io, re
from typing import List

import arxiv
import boto3
import requests
from dateutil import tz
from feedgen.feed import FeedGenerator
from feedparser import parse as fp_parse
from openai import OpenAI
from pydub import AudioSegment

# -------------------------- Config ---------------------------------
LOOKBACK_DAYS = int(os.getenv("ARXIV_LOOKBACK_DAYS", "365"))
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "50"))
CATEGORIES = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ["S3_BUCKET"]
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

VOICE_ID = "Matthew"          # Polly neural voice
SPEECH_RATE = "85%"            # prosody rate
MAX_POLLY_CHARS = 2850         # headroom under 3000 limit

# -------------------------- Helpers --------------------------------

def safe_fetch(url: str, params=None, tries: int = 2, timeout: int = 10):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"⚠️  {url} attempt {attempt+1}/{tries} failed:", e)
            time.sleep(1)
    return None


def build_paper_pool(max_per: int = 25) -> List[dict]:
    """Return a list[dict] of papers from multiple sources."""
    pool: List[dict] = []
    client = arxiv.Client()

    # 1️⃣ arXiv relevance
    arxiv_q = arxiv.Search(query=CATEGORIES, max_results=max_per,
                           sort_by=arxiv.SortCriterion.Relevance)
    arxiv_papers = list(client.results(arxiv_q))
    for r in arxiv_papers:
        pool.append({"title": r.title, "id": r.get_short_id(),
                     "summary": r.summary, "pdf": r.pdf_url})
    print(f"arXiv: {len(arxiv_papers)}")

    # 2️⃣ PapersWithCode trending
    pwc = safe_fetch("https://paperswithcode.com/trending?mod=api&limit=50")
    if pwc:
        for p in pwc[:max_per]:
            pool.append({"title": p["title"],
                         "id": p["arxiv_id"] or p["url"],
                         "summary": p["abstract"],
                         "pdf": p.get("url_pdf")})
    print(f"PapersWithCode: {len(pwc or [])}")

    # 3️⃣ Semantic Scholar
    ss = safe_fetch("https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": "machine learning", "limit": max_per,
                            "year": datetime.date.today().year,
                            "fields": "title,abstract,url"})
    if ss:
        for p in ss.get("data", []):
            pool.append({"title": p["title"], "id": p["paperId"],
                         "summary": p.get("abstract", ""), "pdf": p.get("url")})
    print(f"SemanticScholar: {len(ss.get('data', []) if ss else 0)}")

    # 4️⃣ OpenAlex
    oa = safe_fetch("https://api.openalex.org/works",
                    params={"filter": "concept.id:C41008148", "per_page": max_per})
    if oa:
        count_oa = 0
        for w in oa.get("results", []):
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            pool.append({"title": w.get("title", "Untitled"),
                         "id": w.get("id", str(count_oa)),
                         "summary": "", "pdf": src.get("url", "")})
            count_oa += 1
        print(f"OpenAlex: {count_oa}")
    else:
        print("OpenAlex: 0")

    # 5️⃣ OpenReview (ICLR 2024)
    orv = safe_fetch("https://api.openreview.net/notes",
                     params={"invitation": "ICLR.cc/2024/Conference/-/Blind_Submission",
                             "limit": max_per})
    if orv:
        pool.extend({"title": n["content"]["title"], "id": n["id"],
                     "summary": n["content"]["abstract"],
                     "pdf": n["content"].get("pdf", "")}
                    for n in orv.get("notes", []))
    print(f"OpenReview: {len(orv.get('notes', []) if orv else 0)}")

    if not pool:
        print("⚠️  All sources empty; broad arXiv fallback")
        broad = list(client.results(arxiv.Search(query="machine learning",
                                                 max_results=50,
                                                 sort_by=arxiv.SortCriterion.Relevance)))
        pool = [{"title": r.title, "id": r.get_short_id(),
                 "summary": r.summary, "pdf": r.pdf_url} for r in broad]

    print(f"🥡  Total pool size: {len(pool)}")
    return pool


def pick_paper() -> dict:
    pool = build_paper_pool(MAX_RESULTS)
    # weight arXiv items slightly higher
    weights = [3 if p["id"].isdigit() else 1 for p in pool]
    return random.choices(pool, weights=weights, k=1)[0]

# -------------------------- Text helpers ---------------------------

def scrub(text: str) -> str:
    """Remove leading bullet marks and collapse newlines."""
    lines = [re.sub(r"^[-*•\d]+[.)]?\s*", "", ln).strip()
             for ln in text.splitlines()]
    return " ".join(filter(None, lines))


def chunk_text(text: str, max_len: int = MAX_POLLY_CHARS):
    """Yield <=max_len chunks split on sentence boundaries."""
    while text:
        snippet = text[:max_len]
        cut = snippet.rfind(".")
        if cut == -1 or cut < max_len * 0.6:
            cut = max_len
        yield text[:cut + 1].strip()
        text = text[cut + 1:].lstrip()

# -------------------------- Main -----------------------------------

def main():
    paper = pick_paper()
    paper_id = paper["id"][:30].replace("/", "_")
    paper_title = paper["title"]
    paper_summary = paper["summary"] or "(summary unavailable)"

    print(f"🎙️  Generating episode for {paper_id}: {paper_title}")

    # -------- OpenAI summarisation --------
    openai_client = OpenAI(api_key=OPENAI_KEY)
    prompt_system = (
        "You are a science communicator. Write a ~1300 word script (plain sentences, no bullets) "
        "that explains the paper clearly, tells a story, and ends with one takeaway."
    )
    prompt_user = f"Title: {paper_title}\n\nAbstract: {paper_summary}"

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt_system},
                 {"role": "user", "content": prompt_user}],
        temperature=0.5,
    )
    summary = scrub(resp.choices[0].message.content.strip())

    intro = (
        f"Welcome to AI Paper Snacks, where we dive into the latest research in machine learning. "
        f"Today’s topic: {paper_title}. "
    )
    outro = (
        "That’s all for this episode. Follow the feed for more research breakdowns next week. "
        "Thanks for listening!"
    )
    script = f"{intro} {summary} {outro}"

    # -------- Polly TTS (neural) --------
    polly = boto3.client("polly", region_name=AWS_REGION)
    segments = []
    for chunk in chunk_text(script):
        ssml = f"<speak><amazon:domain name='conversational'>{chunk}</amazon:domain></speak>"
    # --- join & export MP3 ---
    from pydub import AudioSegment
    import io, datetime

    final_audio = AudioSegment.empty()
    for seg in segments:
        final_audio += seg
    buf = io.BytesIO()
    final_audio.export(buf, format="mp3")
    audio_bytes = buf.getvalue()

    # --- upload to S3 ---
    key_mp3 = f"episodes/{paper_id}_{datetime.date.today()}.mp3"
    s3 = boto3.client("s3", region_name=AWS_REGION)
    public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key_mp3}"
    s3.put_object(Bucket=S3_BUCKET, Key=key_mp3, Body=audio_bytes,
                  ACL="public-read", ContentType="audio/mpeg")
    print(f\"📤 Uploaded MP3 to {public_url}\")

    # --- fetch & update RSS ---
    try:
        feed_obj = s3.get_object(Bucket=S3_BUCKET, Key=\"feed.xml\")
        existing_xml = feed_obj[\"Body\"].read()
        parsed = fp_parse(existing_xml)
    except s3.exceptions.NoSuchKey:
        parsed = None

    fg = FeedGenerator()
    fg.load_extension('podcast')
    fg.title('Weekly AI Paper Snacks')
    fg.link(href=f'https://{S3_BUCKET}.s3.amazonaws.com/feed.xml')
    fg.description('AI/ML/RL papers digested into ~10-min audio.')
    fg.language('en-us')

    # copy old items
    if parsed and parsed.entries:
        for e in parsed.entries:
            fe = fg.add_entry()
            fe.id(e.id); fe.title(e.title); fe.description(e.description)
            fe.pubDate(e.published)
            enc = e.enclosures[0]
            fe.enclosure(enc['href'], enc.get('length','0'), enc.get('type','audio/mpeg'))

    # add new item
    fe = fg.add_entry()
    fe.id(paper_id)
    fe.title(paper_title)
    fe.description(script[:200])
    fe.pubDate(datetime.datetime.now(datetime.timezone.utc))
    fe.enclosure(public_url, str(len(audio_bytes)), 'audio/mpeg')

    rss_bytes = fg.rss_str(pretty=True)
    s3.put_object(Bucket=S3_BUCKET, Key='feed.xml', Body=rss_bytes,
                  ACL='public-read', ContentType='application/rss+xml')
    print(\"📝 feed.xml updated\\n🎉 Episode complete!\")


if __name__ == \"__main__\":       # <- this line actually runs main()
    main()
