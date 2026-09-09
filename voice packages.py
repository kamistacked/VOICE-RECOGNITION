from speechbrain.inference.speaker import SpeakerRecognition
import torch

# 1. Load the pre-trained ECAPA-TDNN model from Hugging Face
print("Loading model (downloads automatically on first run)...")
verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir_speaker_model"
)

def check_voices(file1, file2, label):
    print(f"\n--- Testing: {label} ---")
    # verify_files returns: (similarity_score, prediction_boolean)
    score, prediction = verifier.verify_files(file1, file2)

    # score is a PyTorch tensor (cosine similarity between -1 and 1)
    similarity = score.item()
    is_match = bool(prediction[0])

    print(f"Similarity Score : {similarity:.4f}")
    print(f"Result           : {'✅ MATCH (Same Person)' if is_match else '❌ NO MATCH (Different Person)'}")
    return similarity

# Test 1: Same person, two different phrases
check_voices("user_sample1.wav", "user_sample2.wav", "Positive Test (You vs. You)")

# Test 2: You vs. Impostor
check_voices("user_sample1.wav", "impostor_sample.wav", "Negative Test (You vs. Impostor)")