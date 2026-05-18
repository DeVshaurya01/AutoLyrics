import os
import winsound

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nus-smc-corpus_48"))
    lyrics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "interim", "lyrics"))
    
    os.makedirs(lyrics_dir, exist_ok=True)
    
    song_ids = [f"{i:02d}" for i in range(2, 21)]
    
    print("Welcome to the Song Identification Helper!")
    print("This script will play a short sample for each song from 02 to 20.")
    print("Type the lyrics/title for the song and press Enter to save.")
    print("Leave blank and press Enter to skip.\n")
    
    for song_id in song_ids:
        wav_path = None
        # Try to find a wav file for this song ID in any singer's directory
        if os.path.exists(base_dir):
            for singer in os.listdir(base_dir):
                possible_path = os.path.join(base_dir, singer, "sing", f"{song_id}.wav")
                if os.path.exists(possible_path):
                    wav_path = possible_path
                    break
        else:
            print(f"Error: {base_dir} not found.")
            return

        out_file = os.path.join(lyrics_dir, f"{song_id}.txt")
        if os.path.exists(out_file):
            print(f"Song {song_id} already has lyrics. Skipping...")
            continue
            
        if wav_path:
            print(f"\n--- Playing song {song_id} ---")
            print(f"File: {wav_path}")
            
            try:
                # Play audio asynchronously
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                
                lyrics = input(f"Enter lyrics for song {song_id} (or leave blank to skip): ")
                
                # Stop audio
                winsound.PlaySound(None, winsound.SND_PURGE)
                
                if lyrics.strip():
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(lyrics.strip() + "\n")
                    print(f"Saved to {out_file}")
                else:
                    print(f"Skipped song {song_id}.")
            except Exception as e:
                print(f"Error playing audio or saving: {e}")
                winsound.PlaySound(None, winsound.SND_PURGE)
        else:
            print(f"Could not find .wav file for song {song_id}.")

if __name__ == "__main__":
    main()
