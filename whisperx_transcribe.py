import os
from whisperx import load_model
from whisperx.diarize import DiarizationPipeline
from transformers import pipeline

# ─── Load & validate your HF_TOKEN from env ──────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN=hf_<your Hugging Face token> in your environment")
# ───────────────────────────────────────────────────────────────────

# 1) Initialize WhisperX ASR & diarization
asr_model = load_model("small", device="cpu", compute_type="float32")
diarizer  = DiarizationPipeline(use_auth_token=HF_TOKEN)

# 2) Initialize local summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def transcribe_and_diarize(audio_path: str) -> str:
    """Transcribe audio, diarize speakers, write transcript, and return its path."""
    # ASR
    asr_result = asr_model.transcribe(audio_path)
    print("✅ Transcription complete")

    # Diarization
    df = diarizer(audio_path)
    segments = df.to_dict(orient="records")
    print("✅ Diarization complete")

    # Merge speaker labels
    aligned = []
    for seg in asr_result["segments"]:
        spk = "Unknown"
        for d in segments:
            if d["start"] <= seg["start"] < d["end"]:
                spk = d["speaker"]
                break
        seg["speaker"] = spk
        aligned.append(seg)

    # Write speaker-tagged transcript
    transcript_path = "transcript_with_speakers.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in aligned:
            f.write(f"{seg['speaker']}: {seg['text']}\n")
    print(f"✅ Transcript saved to {transcript_path}")

    return transcript_path

def generate_summary_local(transcript_path: str, summary_path: str):
    """Chunk the transcript, summarize each chunk locally, and save the joined summary."""
    text = open(transcript_path, "r", encoding="utf-8").read()

    # Simple fixed-size chunking
    chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
    parts = []
    for c in chunks:
        out = summarizer(c, max_length=150, min_length=50, do_sample=False)
        parts.append(out[0]["summary_text"])

    summary = "\n\n".join(parts)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Local summary written to {summary_path}")

def main():
    audio_file   = "meeting.mp3"             # your input
    transcript   = transcribe_and_diarize(audio_file)
    summary_file = "meeting_summary.txt"     # desired output
    generate_summary_local(transcript, summary_file)

    print("\n🎉 Done! Files generated:")
    print(f" • {transcript}")
    print(f" • {summary_file}")

if __name__ == "__main__":
    main()