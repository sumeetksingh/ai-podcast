"""make_episode.py – Automated AI/ML Podcast Generator
---------------------------------------------------
Publishes a ~10‑minute, intro‑outro podcast episode each run. Sources:
• arXiv (relevance‑sorted)          • PapersWithCode Trending  
• Semantic Scholar search           • OpenAlex concept graph  
• OpenReview conference submissions

High‑level flow
---------------
1. Build a paper pool (up to 5×25 items) with resilient HTTP fetches.  
2. Pick one weighted‑random paper (bias top arXiv hits).  
3. Summarise to ~1 300 words via OpenAI.  
4. Wrap intro/outro lines.  
5. Chunk & TTS with Amazon Polly Neural (Matthew, conversational).  
6. Upload MP3 to S3 → episodes/<id>_<YYYY‑MM‑DD>.mp3.  
7. Rebuild feed.xml (feedparser + feedgen).  

Env vars
--------
AWS_REGION, S3_BUCKET, OPENAI_API_KEY  – required (workflow secrets).  
ARXIV_LOOKBACK_DAYS, ARXIV_MAX_RESULTS – optional tuning (defaults 365/50).
"""

from __future__ import annotations
import os, io, re, random, time, datetime
from typing import List
import requests
import arxiv
from openai import OpenAI
import boto3
from pydub import AudioSegment
from feedgen.feed import FeedGenerator
import feedparser

# ---------------- Config ----------------
AWS_REGION   = os.environ["AWS_REGION"]
S3_BUCKET    = os.environ["S3_BUCKET"]
OPENAI_KEY   = os.environ["OPENAI_API_KEY"]
LOOKBACK_DAYS = int(os.getenv("ARXIV_LOOKBACK_DAYS", "365"))
MAX_RESULTS   = int(os.getenv("ARXIV_MAX_RESULTS", "50"))
VOICE_ID      = "Matthew"            # neural US‑male; change to Olivia etc.
SPEECH_RATE   = "85%"               # slower for ~10‑min runtime

# ------------- Helpers -----------------
client = OpenAI(api_key=OPENAI_KEY)
ARXIV_CLIENT = arxiv.Client()
CATEGORIES = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"


def scrub(text: str) -> str:
    """Remove leading bullets / numbers so Polly won't read them."""
    lines = [re.sub(r"^[-*•\d]+[.)]?\s*", "", ln).strip() for ln in text.splitlines()]
    return " ".join([ln for ln in lines if ln])


def safe_fetch(url: str, params: dict | None = None, timeout=10, tries=2):
    for attempt in range(tries):
        try:
            return requests.get(url, params=params, timeout=timeout).json()
        except Exception as e:
            print(f"⚠️  {url} attempt {attempt+1}/{tries} failed: {e}")
            time.sleep(1)
    return None


# ----------- Paper pool builder --------

def build_paper_pool(max_per: int = 25) -> List[dict]:
    pool: List[dict] = []

    # arXiv relevance list
    q = arxiv.Search(query=CATEGORIES,
                     max_results=max_per,
                     sort_by=arxiv.SortCriterion.Relevance)
    arxiv_papers = list(ARXIV_CLIENT.results(q))
    for r in arxiv_papers:
        pool.append({"title": r.title, "id": r.get_short_id(),
                     "summary": r.summary, "pdf": r.pdf_url})
    print("arXiv:", len(arxiv_papers))

    # PapersWithCode Trending
    pwc = safe_fetch("https://paperswithcode.com/trending?mod=api&limit=50")
    if pwc:
        for p in pwc[:max_per]:
            pool.append({"title": p["title"], "id": p["arxiv_id"] or p["url"],
                         "summary": p["abstract"], "pdf": p.get("url_pdf")})
    print("PapersWithCode:", len(pwc or []))

    # Semantic Scholar top‑cited current year
    ss = safe_fetch("https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": "machine learning", "limit": max_per,
                            "year": datetime.date.today().year,
                            "fields": "title,abstract,url"})
    if ss:
        for p in ss.get("data", []):
            pool.append({"title": p["title"], "id": p["paperId"],
                         "summary": p.get("abstract", ""), "pdf": p.get("url")})
    print("SemanticScholar:", len(ss.get("data", []) if ss else 0))

    # OpenAlex AI concept
    oa = safe_fetch("https://api.openalex.org/works",
                    params={"filter": "concept.id:C41008148", "per_page": max_per})
    if oa:
        for w in oa.get("results", []):
            pool.append({"title": w["title"], "id": w["id"],
                         "summary": "", "pdf": w["primary_location"]["source"].get("url")})
    print("OpenAlex:", len(oa.get("results", []) if oa else 0))

    # OpenReview ICLR‑24
    orv = safe_fetch("https://api.openreview.net/notes",
                     params={"invitation": "ICLR.cc/2024/Conference/-/Blind_Submission",
                             "limit": max_per})
    if orv:
        for n in orv.get("notes", []):
            pool.append({"title": n["content"]["title"], "id": n["id"],
                         "summary": n["content"]["abstract"], "pdf": n["content"]["pdf"]})
    print("OpenReview:", len(orv.get("notes", []) if orv else 0))

    if not pool:
        raise RuntimeError("All sources returned zero papers — network blocked?")

    print("🥡  Total pool size:", len(pool))
    return pool


def pick_paper() -> dict:
    pool = build_paper_pool(MAX_RESULTS)
    weights = [3 if p["id"].isdigit() else 1 for p in pool]  # bias arXiv numeric IDs
    return random.choices(pool, weights=weights, k=1)[0]


# ---------------- Main ------------------

def main():
    paper = pick_paper()
    paper_title   = paper["title"]
    paper_summary = paper["summary"][:4000]
    paper_id      = re.sub(r"[\/\s]", "_", paper["id"])[:40]

    # ----- GPT Summarise -----
    system_prompt = (
        "You are a science communicator. Write a ~1300‑word podcast script (plain sentences) "
        "that hooks the listener, explains the paper clearly, gives practical context, and "
        "ends with a memorable takeaway. Avoid bullet points."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Title: {paper_title}\n\nAbstract: {paper_summary}"}
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=messages,
                                          temperature=0.4)
    summary = scrub(resp.choices[0].message.content.strip())

    intro = (f"Welcome to AI Paper Snacks, where we unpack the most interesting research in "
             f"machine learning. Today’s topic: {paper_title}. ")
    outro = ("Thanks for listening! Follow the feed and join us next week for another deep dive.")
    script = f"{intro} {summary} {outro}"

    # ----- Polly Neural TTS (chunk + stitch) -----
    polly = boto3.client("polly", region_name=AWS_REGION)

    def chunks(txt: str, max_chars: int = 2850):
        while txt:
            part = txt[:max_chars]
            cut  = part.rfind(".")
            if cut == -1 or cut < max_chars * 0.6:
                cut = max_chars
            yield txt[:cut + 1].strip(); txt = txt[cut + 1:].lstrip()

    audio_segments = []
    for chunk in chunks(script):
        ssml = (f"<speak><amazon:domain name='conversational'>"
                f"<prosody rate='{SPEECH_RATE}'>{chunk}</prosody></amazon:domain></speak>")
        rsp = polly.synthesize_speech(Text=ssml, TextType="ssml",
                                      OutputFormat="mp3", VoiceId=VOICE_ID,
                                      Engine="neural")
        audio_segments.append(AudioSegment.from_file(io.BytesIO(rsp["AudioStream"].read()),
                                                     format="mp3"))
    combined = audio_segments[0]
    for seg in audio_segments[1:]:
        combined += seg
    buf = io.BytesIO(); combined.export(buf, format="mp3"); audio_bytes = buf.getvalue()

    today = datetime.date.today().isoformat()
    key_mp3 = f"episodes/{paper_id}_{today}.mp3"
    public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key_mp3}"

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Bucket=S3_BUCKET, Key=key_mp3, Body=audio_bytes,
                  ACL="public-read", ContentType="audio/mpeg")
    print("📤 Uploaded MP3 to", public_url)

    # ----- rebuild feed.xml ----
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="feed.xml")
        old_feed = feedparser.parse(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        old_feed = None

    fg = FeedGenerator(); fg.load_extension('podcast')
    fg.title('AI Paper Snacks')
    fg.link(href=f'https://{S3_BUCKET}.s3.amazonaws.com/feed.xml')
    fg.description('Concise, conversational summaries of recent AI research.')
    fg.language('en-us')

    # copy previous entries
    if old_feed and old_feed.entries:
        for e in old_feed.entries:
            fe = fg.add_entry()
            fe.id(e.id); fe.title(e.title); fe.description(e.description)
            fe.pubDate(e.published); enc = e.enclosures[0]
            fe.enclosure(enc['href'], enc.get('length', '0'), enc.get('type', 'audio/mpeg'))

    # add new episode
    fe = fg.add_entry()
    fe.id(paper_id); fe.title(paper_title)
    fe.description(script.split(".")[0])
    fe.pubDate(datetime.datetime.now(datetime.timezone.utc))
    fe.enclosure(public_url, str(len(audio_bytes)), 'audio/mpeg')

    s3.put_object(Bucket=S3_BUCKET, Key="feed.xml", Body=fg.rss_str(pretty=True),
                  ACL="public-read", ContentType="application/rss+xml")
    print("📝 feed.xml updated")
    print("🎉 Episode complete!")


if __name__ == "__main__":
    main()
