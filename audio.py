import os
import time
import wave
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
# Paste your API key here or set via $env:GEMINI_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

MODEL_ID = "gemini-2.5-flash-preview-tts"
OUTPUT_DIR = "voice_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. THE COMPLETE MASTER CASTING TARGET LIST
# (God the Father is 100% completed with Charon_default.wav!)
# ------------------------------------------------------------------------------
TARGETED_ROLES = [
    # =========================================================================
    # THE TRINITY (Jesus Christ & The Holy Spirit)
    # =========================================================================
    ("Alnilam", "default", "natural, gentle, compassionate, and warm shepherd tone"),
    ("Alnilam", "proclamation", "bold, clear, authoritative teaching tone projecting to crowds"),
    ("Alnilam", "triumphant", "joyful, victorious, and rejoicing in spirit"),
    ("Alnilam", "shout", "powerful, loud, commanding, shouting with divine authority"), # Lazarus come forth! / The Cross
    ("Rasalgethi", "default", "serene, calm, still, quiet divine whispering tone"),

    # =========================================================================
    # THE ADVERSARY (100% Dedicated to Satan, Demons, and Serpent)
    # =========================================================================
    ("Gacrux", "cunning", "subtle, smooth, deceptively quiet, suspicious, and calculating"),

    # =========================================================================
    # THE CELESTIAL HERALD (Gabriel / Annunciation Theophanies)
    # =========================================================================
    ("Achird", "solemn", "noble, majestic, sacred heraldic messenger tone"),

    # =========================================================================
    # THE 4 DEDICATED REFEREES (Narrators: 2 Male, 2 Female)
    # (Puck default & solemn are already done! Aoede default is already done!)
    # =========================================================================
    ("Aoede", "solemn", "gentle, reverent, quiet, and slow female narration"),
    ("Zephyr", "default", "natural, steady, articulate male storytelling narration"),
    ("Zephyr", "solemn", "deep, solemn, reverent, and slow male narration"),
    ("Autonoe", "default", "natural, warm, dignified female historical narration"),
    ("Autonoe", "solemn", "solemn, quiet, reverent female narration"),

    # =========================================================================
    # SPECIALIZED HUMAN EMOTIONS & SHOUTS
    # =========================================================================
    ("Fenrir", "hesitant", "timid, insecure, slow, stammering with quiet hesitation and humble breath"), # Moses at the bush
    ("Fenrir", "shout", "loud, thunderous battle cry and military command"),                           # Joshua at Jericho / Gideon
    ("Zephyr", "astonished", "overwhelmed, breathless, bewildered with holy awe, speaking rapidly without thinking"), # Peter (Transfiguration)
    ("Algieba", "astonished", "shocked, trembling, breathless, and terrified from witnessing sudden disaster"),       # Job's calamity messenger
    ("Algenib", "hesitant", "reluctant, cautious, and hesitant messenger tone"),                      # Fearful spies / servants
    ("Iapetus", "shout", "harsh, angry, aggressive shouting crowd and soldier tone"),                  # The mob: "Crucify Him!" / Acts riots
    ("Kore", "shout", "loud, ringing, bold female victory shout and rallying cry"),                    # Deborah: "Up!" / Elizabeth's loud voice

    # =========================================================================
    # MORTAL ACTORS & ACTRESSES (The Players — Baseline Dialogues)
    # =========================================================================
    ("Enceladus", "default", "weathered, mature elder male tone"),
    ("Enceladus", "solemn", "deep, thoughtful, solemn elder counselor tone"),
    ("Orus", "default", "formal, stern, measured aristocratic male tone"),
    ("Iapetus", "default", "rough, gravelly, heavy warrior male tone"),
    ("Algenib", "default", "sharp, urgent male scout and messenger tone"),
    ("Algieba", "default", "breathless, strained, dramatic male servant tone"),
    ("Schedar", "default", "weary, serious male traveler tone"),
    ("Sulafat", "default", "formal, deliberate male advisor tone"),
    ("Callirrhoe", "default", "warm, mature, maternal, loving female tone"),
    ("Leda", "default", "bright, youthful, clear maiden and daughter tone"),
    ("Despina", "default", "gentle, humble, sincere female tone")
]

# Standard calibration test sentence
TEXT_TO_SPEAK = "The old house stood at the edge of the moor, its windows dark save for a single candle flickering."

def save_wav_file(filename: str, audio_bytes: bytes):
    """Wraps Gemini's 24kHz 16-bit PCM stream into a standard playable WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(24000)   # 24kHz
        wf.writeframes(audio_bytes)

print(f"Targeting {len(TARGETED_ROLES)} specific missing roles into ./{OUTPUT_DIR}/ ...\n")

generated_count = 0
skipped_count = 0

for voice_name, mood_key, mood_desc in TARGETED_ROLES:
    filename = f"{OUTPUT_DIR}/{voice_name}_{mood_key}.wav"

    # RULE 1: Never overwrite or re-download an existing file!
    if os.path.exists(filename):
        print(f"  [Skipped - Exists] {voice_name}_{mood_key}.wav")
        skipped_count += 1
        continue

    print(f"Generating: Voice='{voice_name}' | Mood='{mood_key}' ({mood_desc[:35]}...)...")

    try:
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            )
        )

        if mood_key == "default":
            prompt_content = TEXT_TO_SPEAK
        else:
            prompt_content = f"Say in a {mood_desc} tone: {TEXT_TO_SPEAK}"

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt_content,
            config=config
        )

        audio_bytes = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    audio_bytes = part.inline_data.data
                    break

        if audio_bytes:
            save_wav_file(filename, audio_bytes)
            print(f"  ✓ Saved: {filename}")
            generated_count += 1
        else:
            print(f"  ✗ No audio data returned for {voice_name}_{mood_key}")

        # Polite 6-second delay between requests to preserve API limits
        time.sleep(6)

    except Exception as e:
        print(f"  ⚠️ Error generating {voice_name}_{mood_key}: {e}")
        print("  --> Pausing for 15s to allow API cooldown...")
        time.sleep(15)

print("\n==================================================")
print(f" RUN COMPLETE!")
print(f" Newly Generated: {generated_count}")
print(f" Already Existed (Skipped): {skipped_count}")
print(f" Total Library Size: {len(os.listdir(OUTPUT_DIR))} files in ./{OUTPUT_DIR}/")
print("==================================================")
