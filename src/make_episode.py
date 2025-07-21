#!/usr/bin/env python3
"""
make_episode.py  –  Multi-source AI/ML podcast generator
========================================================
Every run (manual or cron) will:
1. Build a paper pool (≤5×25 items) from:
     • arXiv            • PapersWithCode “Trending”
     • Semantic Scholar • OpenAlex       • OpenReview (ICLR 2024)
2. Pick one weighted-random popular paper.
3. Ask OpenAI for ~1 300-word summary; add intro/outro.
4. Synthesize ~10-min audio with Amazon Polly Neural (Matthew, conversational),
   chunking >3 000-char scripts automatically.
5. Upload episodes/<paperid>_<YYYY-MM-DD>.mp3 to S3 and rewrite feed.xml.
No queue/state files – every run publishes a fresh episode.

Requires secrets / env:
  OPENAI_API_KEY  AWS_ACCESS_KEY_ID  AWS_SECRET_ACCESS_KEY
  S3_BUCKET       AWS_REGION (default us-east-1)

Dependencies (requirements.txt):
  openai>=1.23.0  boto3>=1.34.0  requests==2.31.0
  feedgen>=1.0.0  feedparser>=6.0.10  arxiv==2.1.0
  python-dateutil  pydub>=0.25.1  ffmpeg in PATH
"""
from __future__ import annotations

import datetime, io, os, random, re, time
from typing import List

import arxiv
import boto3
import requests
from dateutil import tz
from feedgen.feed import FeedGenerator
from feedparser import parse as fp_parse
from openai import OpenAI
from pydub import AudioSegment

# ---------------- Configuration ------------------------------------
LOOKBACK_DAYS   = int(os.getenv("ARXIV_LOOKBACK_DAYS", "365"))
MAX_RESULTS     = int(os.getenv("ARXIV_MAX_RESULTS", "50"))
CATEGORIES      = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"

AWS_REGION      = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET       = os.environ["S3_BUCKET"]
OPENAI_KEY      = os.environ["OPENAI_API_KEY"]

VOICE_ID        = "Matthew"      # Polly neural voice (try Olivia, Amy…)
SPEECH_RATE     = "85%"          # <prosody rate="">
MAX_POLLY_CHARS = 2850           # leave headroom under 3000 limit
INTRO_TMPL      = (
    "Welcome to AI Paper Snacks, where we dive into machine-learning "
    "research one paper at a time. Today’s topic: {title}. "
)
OUTRO_TEXT      = (
    "That’s it for this episode. Join us next week for another research "
    "breakdown. Thanks for listening!"
)

# -------------------------------------------------------------------
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
    """Collect papers from multiple sources; each dict has title,id,summary,pdf."""
    pool: List[dict] = []
    client = arxiv.Client()

    # 1️⃣ arXiv relevance --------------------------------------------------
    arxiv_q = arxiv.Search(query=CATEGORIES, max_results=max_per,
                           sort_by=arxiv.SortCriterion.Relevance)
    for r in client.results(arxiv_q):
        pool.append({"title": r.title, "id": r.get_short_id(),
                     "summary": r.summary, "pdf": r.pdf_url})
    print(f"arXiv: {len(pool)}")

    # 2️⃣ PapersWithCode ---------------------------------------------------
    pwc = safe_fetch("https://paperswithcode.com/trending?mod=api&limit=50")
    if pwc:
        for p in pwc[:max_per]:
            pool.append({"title": p["title"],
                         "id": p["arxiv_id"] or p["url"],
                         "summary": p["abstract"],
                         "pdf": p.get("url_pdf")})
    print(f"PapersWithCode: {len(pwc or [])}")

    # 3️⃣ Semantic Scholar -------------------------------------------------
    ss = safe_fetch("https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": "machine learning", "limit": max_per,
                            "year": datetime.date.today().year,
                            "fields": "title,abstract,url"})
    ss_items = ss.get("data", []) if ss else []
    for p in ss_items:
        pool.append({"title": p["title"], "id": p["paperId"],
                     "summary": p.get("abstract", ""), "pdf": p.get("url")})
    print(f"SemanticScholar: {len(ss_items)}")

    # 4️⃣ OpenAlex ---------------------------------------------------------
    oa = safe_fetch("https://api.openalex.org/works",
                    params={"filter": "concept.id:C41008148", "per_page": max_per})
    oa_items = oa.get("results", []) if oa else []
    for w in oa_items:
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        pool.append({"title": w.get("title", "Untitled"),
                     "id": w.get("id", str(len(pool))),
                     "summary": "", "pdf": src.get("url", "")})
    print(f"OpenAlex: {len(oa_items)}")

    # 5️⃣ OpenReview (ICLR 2024) ------------------------------------------
    orv = safe_fetch("https://api.openreview.net/notes",
                     params={"invitation": "ICLR.cc/2024/Conference/-/Blind_Submission",
                             "limit": max_per})
    or_items = orv.get("notes", []) if orv else []
    for n in or_items:
        pool.append({"title": n["content"]["title"], "id": n["id"],
                     "summary": n["content"]["abstract"],
                     "pdf": n["content"].get("pdf", "")})
    print(f"OpenReview: {len(or_items)}")

    # Broad arXiv fallback -------------------------------------------------
    if not pool:
        print("⚠️  All sources empty; broad arXiv fallback")
        broad = list(client.results(
            arxiv.Search(query="machine learning", max_results=50,
                         sort_by=arxiv.SortCriterion.Relevance)))
        pool = [{"title": r.title, "id": r.get_short_id(),
                 "summary": r.summary, "pdf": r.pdf_url} for r in broad]

    print(f"🥡  Total pool size: {len(pool)}")
    return pool


def pick_paper() -> dict:
    pool = build_paper_pool(MAX_RESULTS)
    weights = [3 if p["id"].isdigit() else 1 for p in pool]  # bias arXiv
    return random.choices(pool, weights=weights, k=1)[0]

# ---------------- Text utilities -----------------------------------
def scrub(text: str) -> str:
    lines = [re.sub(r"^[-*•\d]+[.)]?\s*", "", ln).strip()
             for ln in text.splitlines()]
    return " ".join(filter(None, lines))

def chunk_text(text: str, max_len: int = MAX_POLLY_CHARS):
    while text:
        snippet = text[:max_len]
        cut = snippet.rfind(".")
        if cut == -1 or cut < max_len * 0.6:
            cut = max_len
        yield text[:cut + 1].strip()
        text = text[cut + 1:].lstrip()

# ---------------- Main ---------------------------------------------
def main():
    paper = pick_paper()
    paper_id = paper["id"][:30].replace("/", "_")
    paper_title = paper["title"]
    paper_summary = paper["summary"] or "(summary unavailable)"

    print(f"🎙️  Generating episode for {paper_id}: {paper_title}")

    # -------- OpenAI summary --------
    openai_client = OpenAI(api_key=OPENAI_KEY)
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": ("You are a brilliant science communicator. "
                         "Write a clear, story-like ~1300-word script "
                         "with no bullet marks.")},
            {"role": "user",
             "content": f"Title: {paper_title}\n\nAbstract: {paper_summary}"}
        ],
        temperature=0.5,
    )
    summary = scrub(resp.choices[0].message.content.strip())
    script  = f"{INTRO_TMPL.format(title=paper_title)} {summary} {OUTRO_TEXT}"

    # -------- Polly TTS --------
    polly = boto3.client("polly", region_name=AWS_REGION)
    audio_segments = []
    for chunk in chunk_text(script):
        ssml = (f"<speak><amazon:domain name='conversational'>"
                f"<prosody rate='{SPEECH_RATE}'>{chunk}</prosody>"
                f"</amazon:domain></speak>")
        tts = polly.synthesize_speech(
            Text=ssml, TextType="ssml",
            OutputFormat="mp3", VoiceId=VOICE_ID, Engine="neural"
        )
        audio_segments.append(AudioSegment.from_file(
            io.BytesIO(tts["AudioStream"].read()), format="mp3"))

    final_audio = sum(audio_segments[1:], audio_segments[0])
    buf = io.BytesIO()
    final_audio.export(buf, format="mp3")
    audio_bytes = buf.getvalue()

    # -------- Upload MP3 --------
    today_str = datetime.date.today().isoformat()
    key_mp3 = f"episodes/{paper_id}_{today_str}.mp3"
    public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key_mp3}"
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Bucket=S3_BUCKET, Key=key_mp3, Body=audio_bytes,
                  ACL="public-read", ContentType="audio/mpeg")
    print(f"📤 Uploaded MP3 to {public_url}")

    # -------- Update RSS --------
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="feed.xml")
        parsed = fp_parse(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        parsed = None

    fg = FeedGenerator()
    fg.load_extension('podcast')
    fg.title("AI Paper Snacks")
    fg.link(href=f"https://{S3_BUCKET}.s3.amazonaws.com/feed.xml")
    fg.description("AI/ML/RL papers digested into ~10-min audio.")
    fg.language("en-us")
    fg.podcast.itunes_author("Sumeet Singh")
    fg.podcast.itunes_owner(name="Sumeet Kumar Singh", email="sumeetkumarsingh@gmail.com")
    fg.podcast.itunes_image(f"https://ai-podcast-audio-sks.s3.us-east-2.amazonaws.com/ai.jpg")

    if parsed and parsed.entries:
        for e in parsed.entries:
            fe = fg.add_entry()
            fe.id(e.id); fe.title(e.title); fe.description(e.description)
            fe.pubDate(e.published)
            enc = e.enclosures[0]
            fe.enclosure(enc['href'], enc.get('length', '0'),
                         enc.get('type', 'audio/mpeg'))

    fe = fg.add_entry()
    fe.id(paper_id); fe.title(paper_title)
    fe.description(script[:200])
    fe.pubDate(datetime.datetime.now(tz.UTC))
    fe.enclosure(public_url, str(len(audio_bytes)), 'audio/mpeg')

    s3.put_object(Bucket=S3_BUCKET, Key="feed.xml",
                  Body=fg.rss_str(pretty=True), ACL="public-read",
                  ContentType="application/rss+xml")
    print("📝 feed.xml updated\n🎉 Episode complete!")


if __name__ == "__main__":
    main()
