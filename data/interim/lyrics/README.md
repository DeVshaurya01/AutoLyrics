# Canonical Song Lyrics (NUS-48E)

Each file (01.txt – 20.txt) must contain the COMPLETE lyrics for that song,
space-delimited or one word per line. These are used to reconstruct
word-level transcriptions from the phone-level label files.

**Format**: plain text, lowercase, no punctuation except apostrophes.
Each file maps to the song numbered in the NUS-48E corpus.

## How to identify each song

The NUS-48E paper does not publish a song-ID → title mapping. To identify
each song, **listen to one .wav per song ID**. Any singer who recorded
that song will work — e.g. for song 02, try playing the first available:

    data/raw/nus-smc-corpus_48/*/sing/02.wav

## Confirmed so far

| File   | Song Title    | How identified                                 |
|--------|---------------|------------------------------------------------|
| 01.txt | Edelweiss     | Phone sequence "ey d ah l v ay s" in 01.txt    |
| 02.txt | ?             | Listen to `*/sing/02.wav` and fill in          |
| 03.txt | ?             | Listen to `*/sing/03.wav` and fill in          |
| ...    | ?             |                                                |
| 20.txt | ?             | Listen to `*/sing/20.wav` and fill in          |

## Tip: not every singer sings every song

Each singer recorded only 4 songs. So `ADIZ/sing/` only has songs `01, 09,
13, 18` for example. To find a recording of song 05, look across all
singers — at least one will have it.

## Workflow

1. Open File Explorer at `data/raw/nus-smc-corpus_48/`
2. For each song ID 01–20, play one of its `.wav` files
3. Identify the song, find the lyrics online
4. Paste lyrics into `<song_id>.txt`, lowercase, no punctuation except `'`
5. When all 20 are done, run `python scripts/prepare_data.py`

Example for `01.txt` (Edelweiss):
```
edelweiss edelweiss every morning you greet me small and white clean and bright you look happy to meet me blossom of snow may you bloom and grow bloom and grow forever edelweiss edelweiss bless my homeland forever
```
