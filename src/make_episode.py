#!/usr/bin/env python3
"""
Generate one “paper-of-the-week” podcast episode:

1. Pick next arXiv paper from papers/queue.json (round-robin via papers/state.json)
2. Fetch title & abstract with arxiv API
3. Ask OpenAI (GPT-4o-mini) for a ~6-minute, lay-friendly summary
4. Convert summary to MP3 with Amazon Polly
5. Upload MP3 to S3 and update feed.xml (also in S3)
6. Advance pointer in papers/state.json so next run picks the following paper
"""

import os, json, io, datetime, tempfile
from pathlib import Path

import arxiv                     # pip install arxiv
import openai                    # pip install openai
import boto3                     # pip install boto3
from feedgen.feed import FeedGenerator   # pip install feedgen
from dateutil import tz          # pip install python-dateutil

# ---------- local paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
QUEUE_FILE = BASE_DIR / "papers" / "queue.json"
STATE_FILE = BASE_DIR / "papers" / "state.json"

# ---------- env vars coming from GitHub Secrets ----------
AWS_REGION   = os.environ["AWS_REGION"]
S3_BUCKET    = os.environ["S3_BUCKET"]
OPENAI_KEY   = os.environ["OPENAI_API_KEY"]

# ---------- step 1: pick the next paper ----------
queue = json.loads(Path(QUEUE_FILE).read_text())
state = json.loads(Path(STATE_FILE).read_text())
idx   = state.get("next_index", 0) % len(queue)

paper_id   = queue[idx]["id"]
paper_meta = next(arxiv.Search(id_list=[paper_id]).results())
paper_txt  = f"{paper_meta.title}\n\n{paper_meta.summary}"

print(f"🎙️  Generating episode for {paper_id}: {paper_meta.title}")

# ---------- step 2: summarise with ChatGPT ----------
from openai import OpenAI

client = OpenAI(api_key=OPENAI_KEY)
msg = [
    {"role": "system",
     "content": ("You are a brilliant science communicator. "
                 "Produce a clear, story-style, 6-minute audio script "
                 "explaining the paper to a general tech audience. "
                 "Open with a hook, end with one key takeaway.")},
    {"role": "user", "content": paper_txt}
]

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=msg,
    temperature=0.5,
)
summary = resp.choices[0].message.content.strip()
print(f"✅ Got summary ({len(summary.split())} words)")

# ---------- step 3: text-to-speech ----------
polly = boto3.client("polly", region_name=AWS_REGION)
tts   = polly.synthesize_speech(
            Text=summary,
            OutputFormat="mp3",
            VoiceId="Joanna",             # change voice here if you like
            Engine="neural"
       )

audio_bytes = tts["AudioStream"].read()
key_mp3     = f"episodes/{paper_id}.mp3"
public_url  = f"https://{S3_BUCKET}.s3.amazonaws.com/{key_mp3}"

s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------- step 4: upload MP3 ----------
s3.put_object(
    Bucket=S3_BUCKET,
    Key=key_mp3,
    Body=audio_bytes,
    ACL="public-read",
    ContentType="audio/mpeg"
)
print(f"📤 Uploaded MP3 to {public_url}")

# ---------- step 5: fetch + update RSS feed ----------
feed_obj = s3.get_object(Bucket=S3_BUCKET, Key="feed.xml")
xml      = feed_obj["Body"].read()

fg = FeedGenerator()
fg.load_extension('podcast')     # optional but enables iTunes tags
fg.parse(xml)                    # parse existing feed

fe = fg.add_entry()
fe.id(paper_id)
fe.title(paper_meta.title)
fe.enclosure(public_url, str(len(audio_bytes)), "audio/mpeg")
fe.pubDate(datetime.datetime.now(tz.tzutc()))
fe.description(summary.split("\n")[0])        # first sentence as teaser

rss_bytes = fg.rss_str(pretty=True)
s3.put_object(
    Bucket=S3_BUCKET,
    Key="feed.xml",
    Body=rss_bytes,
    ACL="public-read",
    ContentType="application/rss+xml"
)
print("📝 feed.xml updated")

# ---------- step 6: advance pointer ----------
state["next_index"] = idx + 1
STATE_FILE.write_text(json.dumps(state, indent=2))
print("🔄 state.json advanced")

print("🎉 Episode complete!")
