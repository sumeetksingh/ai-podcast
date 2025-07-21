# make_episode.py – fully self‑contained weekly podcast generator
# ---------------------------------------------------------------
# Features
#   • Picks a popular AI/ML/CL paper from arXiv (relevance‑sorted, look‑back window)
#   • Summarises it with OpenAI, adds intro/outro
#   • Converts to ~10‑min audio using Amazon Polly Neural voice
#   • Uploads MP3 to S3 and refreshes feed.xml (no fg.parse dependency)
#   • Every workflow run – manual or scheduled – publishes a new episode

from __future__ import annotations

import os, io, re, random, datetime, textwrap
from pathlib import Path
from typing import List

import boto3
from pydub import AudioSegment
import feedparser                          # read existing feed
from feedgen.feed import FeedGenerator     # write new feed
from openai import OpenAI
import arxiv

# ------------------------------------------------------------------
# 0.  Configuration via environment variables / workflow secrets
# ------------------------------------------------------------------
OPENAI_KEY           = os.environ["OPENAI_API_KEY"]
S3_BUCKET            = os.environ["S3_BUCKET"]
AWS_REGION           = os.getenv("AWS_REGION", "us-east-1")
VOICE_ID             = os.getenv("VOICE_ID", "Matthew")      # Neural male US
LOOKBACK_DAYS        = int(os.getenv("ARXIV_LOOKBACK_DAYS", "365"))
MAX_RESULTS          = int(os.getenv("ARXIV_MAX_RESULTS", "50"))
SCRIPT_WORD_TARGET   = int(os.getenv("SCRIPT_WORDS", "1300"))

client = OpenAI(api_key=OPENAI_KEY)
polly  = boto3.client("polly", region_name=AWS_REGION)
s3     = boto3.client("s3",  region_name=AWS_REGION)

# ------------------------------------------------------------------
# 1.  Utility helpers
# ------------------------------------------------------------------

from typing import List
import arxiv, random, datetime, os

# -------------------------------------------------------------------
#  Pick a popular AI/ML paper, widening the date window until one is found
# -------------------------------------------------------------------
LOOKBACK_DAYS = int(os.getenv("ARXIV_LOOKBACK_DAYS", "365"))
MAX_RESULTS   = int(os.getenv("ARXIV_MAX_RESULTS", "50"))
CATEGORIES    = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML"
CLIENT        = arxiv.Client()

def pick_paper() -> arxiv.Result:
    step = 30                         # widen by 30-day increments
    back  = LOOKBACK_DAYS
    while back <= 730:                # cap at 2 years
        date_range = f"[NOW-{back}DAY TO NOW]"
        query = f"{CATEGORIES} AND submittedDate:{date_range}"
        search = arxiv.Search(
            query=query,
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers: List[arxiv.Result] = list(CLIENT.results(search))
        if papers:
            # bias toward top 10 but keep variety
            weights = [max(1, 20-i) for i in range(len(papers))]
            return random.choices(papers, weights=weights, k=1)[0]
        back += step                  # widen and try again
    raise RuntimeError("No papers found in the last 2 years")



def scrub(text: str) -> str:
    """Remove bullets / list markers so Polly won't read them verbatim."""
    cleaned = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*[-*•\d]+[.)]?\s*", "", ln).strip()
        if ln:
            cleaned.append(ln)
    return " ".join(cleaned)


# For Polly: split SSML into <3K char chunks (Neural limit 10K, but safe)
MAX_CHARS = 2850

def yield_chunks(text: str):
    while text:
        snippet = text[:MAX_CHARS]
        split_at = snippet.rfind('.')
        if split_at == -1 or split_at < MAX_CHARS * 0.6:
            split_at = MAX_CHARS
        yield text[:split_at + 1].strip()
        text = text[split_at + 1:].lstrip()

# ------------------------------------------------------------------
# 2.  Main flow
# ------------------------------------------------------------------

def main() -> None:
    paper = pick_paper()
    paper_id  = paper.get_short_id()
    title     = paper.title.strip().replace('\n', ' ')
    abstract  = paper.summary.replace('\n', ' ')
    print(f"📰 Picked paper {paper_id}: {title}")

    # ----- 2a. Summarise with OpenAI -----
    SYSTEM_PROMPT = (
        "You are a brilliant science communicator. "
        f"Write a ~{SCRIPT_WORD_TARGET}-word podcast script (no bullet marks, plain sentences) "
        "that hooks the listener, explains the methods, key results, and practical implications, "
        "and ends with one memorable takeaway."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": abstract}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
    )
    summary = resp.choices[0].message.content.strip()

    # ----- 2b. Assemble full script with intro/outro -----
    intro  = (
        f"Welcome to AI Paper Snacks, where we dive into research that shapes the future. "
        f"Today’s topic: {title}. "
    )
    outro  = (
        "That’s all for this episode. Follow the feed and join us next week for another deep dive. "
        "Thanks for listening!"
    )
    script = scrub(f"{intro} {summary} {outro}")
    word_count = len(script.split())
    print(f"✅ Script ready ({word_count} words)")

    # ----- 2c. Text‑to‑speech with Neural Polly -----
    audio_segments = []
    for chunk in yield_chunks(script):
        ssml = (
            "<speak><amazon:domain name='conversational'>" + chunk + "</amazon:domain></speak>"
        )
        tts = polly.synthesize_speech(
            Text=ssml,
            TextType="ssml",
            OutputFormat="mp3",
            VoiceId=VOICE_ID,
            Engine="neural",
        )
        audio_segments.append(
            AudioSegment.from_file(io.BytesIO(tts["AudioStream"].read()), format="mp3")
        )
    combined = audio_segments[0]
    for seg in audio_segments[1:]:
        combined += seg
    buf = io.BytesIO()
    combined.export(buf, format="mp3")
    audio_bytes = buf.getvalue()

    # filename: <id>_YYYY-MM-DD.mp3
    today_str = datetime.date.today().isoformat()
    key_mp3   = f"episodes/{paper_id}_{today_str}.mp3"
    public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key_mp3}"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key_mp3,
        Body=audio_bytes,
        ACL="public-read",
        ContentType="audio/mpeg",
    )
    print(f"📤 Uploaded MP3 to {public_url}")

    # ----- 2d. Refresh feed.xml (read with feedparser) -----
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key="feed.xml")
        parsed = feedparser.parse(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        parsed = None

    fg = FeedGenerator()
    fg.load_extension('podcast')
    fg.title('AI Paper Snacks')
    fg.link(href=f'https://{S3_BUCKET}.s3.amazonaws.com/feed.xml')
    fg.description('10‑minute audiocasts of impactful AI/ML research')
    fg.language('en-us')

    if parsed and parsed.entries:
        for entry in parsed.entries:
            fe_old = fg.add_entry()
            fe_old.id(entry.id)
            fe_old.title(entry.title)
            fe_old.description(entry.description)
            fe_old.pubDate(entry.published)
            enc = entry.enclosures[0]
            fe_old.enclosure(enc['href'], enc.get('length', '0'), enc.get('type', 'audio/mpeg'))

    fe = fg.add_entry()
    fe.id(paper_id)
    fe.title(title)
    fe.description(script[:200] + '…')
    fe.pubDate(datetime.datetime.now(datetime.timezone.utc))
    fe.enclosure(public_url, str(len(audio_bytes)), 'audio/mpeg')

    rss_bytes = fg.rss_str(pretty=True)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="feed.xml",
        Body=rss_bytes,
        ACL="public-read",
        ContentType="application/rss+xml",
    )
    print("📝 feed.xml updated")
    print("🎉 Episode complete!")


if __name__ == "__main__":
    main()
