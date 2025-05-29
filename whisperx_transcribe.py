import os
from pydub import AudioSegment
from openai import OpenAI, OpenAIError

from whisperx import (
    load_model,
    load_align_model,
    align,
    assign_word_speakers
)
from whisperx.diarize import DiarizationPipeline

# ────────────────────────────────────────────────────────────────
HF_TOKEN       = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not HF_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Please set both HF_TOKEN and OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
# ────────────────────────────────────────────────────────────────

def prepare_audio(in_path: str, out_path: str = "meeting_prepared.wav") -> str:
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    print(f"✅ Audio → {out_path}")
    return out_path

def punctuate(text: str) -> str:
    """Restore punctuation via GPT-3.5-turbo using the new client API."""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a punctuation assistant."},
                {"role": "user",   "content": "Punctuate this without changing any words:\n\n" + text}
            ],
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        print("⚠️ Punctuation API error:", e)
        return text

def transcribe_clean_and_save(audio_file: str) -> str:
    # 1️⃣ Prepare WAV
    wav = prepare_audio(audio_file)

    # 2️⃣ WhisperX ASR
    asr = load_model("small", device="cpu", compute_type="float32")
    result = asr.transcribe(wav)
    segments = result["segments"]
    info     = result.get("language", "en")
    print("✅ ASR done")

    # 3️⃣ Choose language code
    lang = info if isinstance(info, str) else info.get("language", "en")
    print(f"Detected language: {lang}")

    # 4️⃣ Word-level alignment
    align_model, metadata = load_align_model(lang, device="cpu")
    aligned = align(
        segments,
        align_model,
        metadata,
        wav,
        device="cpu"
    )
    print("✅ Word‐level alignment done")

    # 5️⃣ Diarization
    diarizer = DiarizationPipeline(use_auth_token=HF_TOKEN)
    diarize_df = diarizer(wav)
    print("✅ Diarization done")

    # 6️⃣ Attach speaker tags
    asr_with_spk = assign_word_speakers(diarize_df, aligned)

    # 7️⃣ Flatten to list of (speaker, word)
    words = []
    for seg in asr_with_spk["segments"]:
        for w in seg.get("words", []):
            if "speaker" in w and "word" in w:
                words.append((w["speaker"], w["word"]))

    # 8️⃣ Group runs by speaker
    runs = []
    if words:
        cur, buf = words[0][0], [words[0][1]]
        for spk, wd in words[1:]:
            if spk == cur:
                buf.append(wd)
            else:
                runs.append((cur, " ".join(buf)))
                cur, buf = spk, [wd]
        runs.append((cur, " ".join(buf)))

    # 9️⃣ Map raw IDs → Speaker 1…N
    spk_map, idx = {}, 1
    for raw, _ in runs:
        if raw not in spk_map:
            spk_map[raw] = f"Speaker {idx}"
            idx += 1

    # 🔟 Write cleaned, punctuated transcript
    transcript_path = "transcript_cleaned.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for raw, text in runs:
            p = punctuate(text)
            f.write(f"{spk_map[raw]}: {p}\n\n")
    print(f"✅ Transcript → {transcript_path}")
    return transcript_path

def summarize_transcript(in_path: str, out_path: str):
    txt = open(in_path, "r", encoding="utf-8").read()
    prompt = (
        "You are a meeting-note assistant. Given the following speaker-tagged transcript,\n"
        "produce three clearly labeled sections:\n\n"
        "Overview:\n[A concise 1–2 sentence summary]\n\n"
        "Key Points:\n- …\n\n"
        "Next Steps:\n- …\n\n"
        "Transcript:\n\"\"\"\n"
        + txt +
        "\n\"\"\"\n"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"You summarize meeting transcripts."},
            {"role":"user",  "content":prompt}
        ],
        temperature=0.0
    )
    summary = resp.choices[0].message.content.strip()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Summary → {out_path}")

def main():
    transcript_fp = transcribe_clean_and_save("meeting.mp3")
    summarize_transcript(transcript_fp, "meeting_summary.txt")
    print("\n🎉 All done! Files:")
    print(" •", transcript_fp)
    print(" • meeting_summary.txt")

if __name__ == "__main__":
    main()