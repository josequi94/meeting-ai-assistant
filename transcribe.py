from faster_whisper import WhisperModel

# Load the Whisper model (small, accurate, fast)
model = WhisperModel("small")

# Transcribe a local audio file (e.g. meeting.mp3 or .wav)
segments, info = model.transcribe("meeting.mp3")

print("Detected language:", info.language)

# Save the full transcription to a file
with open("transcript.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n")

print("✅ Transcription complete! Saved to transcript.txt.")