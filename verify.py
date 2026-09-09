from speechbrain.inference.speaker import SpeakerRecognition

print("Loading AI model...")
verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir_speaker_model"
)

# The filenames must be inside the quotes:
audio_file_1 = "Record(1) (online-audio-converter.com).wav"
audio_file_2 = "Record(2) (online-audio-converter.com).wav"

print(f"\nComparing '{audio_file_1}' vs '{audio_file_2}'...")

# Compare the two voices
score, prediction = verifier.verify_files(audio_file_1, audio_file_2)

similarity = score.item()
is_match = bool(prediction[0])

print("\n--- RESULTS ---")
print(f"Similarity Score : {similarity:.4f} (1.0 is identical)")

if is_match:
    print("✅ VERIFIED: Same speaker detected!")
else:
    print("❌ FAILED: Different speakers detected!")