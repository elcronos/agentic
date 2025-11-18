# GenAIVideoGenerationVeo3

[![Watch the video]()](/videos/kangaroo_paris/vlog.mp4)

This project shows how to use **Gemini 2.5 + Veo 3** to generate a short, multi-scene video (e.g. a vlog) from a single idea.

The script:

1. **Generates scene prompts** with a text-only Gemini model.
2. **Generates a short clip per scene** with the **Veo 3** video model.
3. **Merges all clips** into one final video using **MoviePy**.

The default example creates a vlog of a **kangaroo visiting Paris for the first time**.

---

## Features

- End-to-end video generation from a single high-level idea.
- Structured **scene planning** using a rich prompt and JSON schema (`Scene` / `SceneResponse` via `pydantic`).
- Per-scene generation using `veo-3.0-generate-preview`.
- Automatic merge of all scene clips into a single `vlog.mp4`.
- Easy to customize:
  - idea / topic
  - main character description
  - personality traits
  - number of scenes
  - aspect ratio
  - camera style and video style

---

## Requirements

- Python 3.10+ (recommended)
- A Google GenAI API key with access to:
  - `gemini-2.5-pro`
  - `veo-3.0-generate-preview`

Python dependencies:

- `google-genai`
- `moviepy`
- `pydantic`
- (optionally) `uv` for fast installs

Install with `uv` (as in the original example):

```bash
uv pip install moviepy google-genai pydantic
```

Or with standard `pip`:

```bash
pip install moviepy google-genai pydantic
```

---

## Environment Setup

Set your Google GenAI API key so `google.genai.Client()` can authenticate.

Typically this is done via an environment variable (check the `google-genai` docs for the exact name used in your version). A common pattern is:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

You can also use a `.env` file or your platform’s secret manager, as long as the environment variable is available when you run the script.

---

## Project Structure

A minimal structure for this project might look like:

```text
GenAIVideoGenerationVeo3/
  ├─ main.py              # Contains generate_scenes, generate_video, merge_videos, generate_vlog
  ├─ scenes/              # (optional) Where scene markdown can be written
  ├─ videos/              # Default output folder for clips + merged vlog
  └─ README.md
```

> If you use a different script name, just adjust the commands below accordingly.

---

## How It Works

### 1. Scene generation (`generate_scenes`)

```python
scenes = generate_scenes(
    idea=idea,
    character_description=character_description,
    character_characteristics="sarcastic, dramatic, emotional, and lovable",
    number_of_scenes=4,
    video_type="vlog",
    video_characteristics="realistic, 4k, cinematic",
    camera_angle="front",
    output_dir="scenes",
)
```

- Builds a long, structured prompt for **Gemini 2.5 Pro** describing:
  - Video topic / idea
  - Main character description
  - Personality
  - Video style and camera angles
- Asks Gemini to return **valid JSON** following the `SceneResponse` schema:
  - `scenes: list[Scene]`
  - Each `Scene` has:
    - `description`: the actual Veo prompt
    - `negative_description`: what to avoid in the clip
- Parses the JSON via `SceneResponse.model_validate_json(...)`.
- Writes a human-readable `scenes.md` (one section per scene).
- Returns a `list[Scene]`.

### 2. Video generation (`generate_video`)

```python
video_file = generate_video(
    prompt=scene.description,
    negative_prompt=scene.negative_description,
    aspect_ratio="16:9",
    output_dir="videos",
    fname="video_0.mp4",
)
```

- Adds some **video rules** (no subtitles, specific aspect ratio, short & cinematic).
- Calls `client.models.generate_videos` with:
  - model: `"veo-3.0-generate-preview"`
  - prompt: the scene description + rules
  - config: `GenerateVideosConfig` (aspect ratio, allowed people, negative_prompt)
- Polls until the video operation is finished.
- Downloads the generated file using `client.files.download(...)` and saves to `output_dir/fname`.
- Returns the local file path.

### 3. Merging clips (`merge_videos`)

The project uses MoviePy to stitch all generated scene clips into a single vlog.

Core function:

```python
import os
from moviepy import VideoFileClip, concatenate_videoclips

def merge_videos(
    video_files: list[str],
    output_file: str = "vlog.mp4",
    output_dir: str = "videos",
) -> str:
    """Merges multiple video files into a single video.

    Args:
        video_files: A list of paths to the video files to merge.
        output_file: The filename for the final merged video.
        output_dir: The directory to save the final video.

    Returns:
        The file path of the merged video.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load each video clip
    clips = [VideoFileClip(file) for file in video_files]

    # Concatenate the video clips
    final_clip = concatenate_videoclips(clips)

    # Write the final video file
    final_clip.write_videofile(
        os.path.join(output_dir, output_file),
        codec="libx264",
        audio_codec="aac",
    )

    return os.path.join(output_dir, output_file)
```

Standalone usage example (e.g. if you already have `video_0.mp4`–`video_3.mp4` in the current folder):

```python
merge_videos(
    ["video_0.mp4", "video_1.mp4", "video_2.mp4", "video_3.mp4"],
    "vlog.mp4",
    output_dir=".",
)
```

In the full pipeline, `generate_vlog` calls `merge_videos` automatically with the list of generated scene files.

### 4. High-level orchestration (`generate_vlog`)

```python
generate_vlog(
    idea="Tourist in Paris for the first time seeing all of the best places",
    character_description="Kangaroo",
    character_characteristics="funny",
    video_type="vlog",
    video_characteristics="realistic, 4k, high quality, vlog",
    camera_angle="front, close-up speaking into the camera",
    aspect_ratio="16:9",
    number_of_scenes=4,
    output_dir="paris_2",
)
```

- Creates `output_dir` if needed.
- Calls `generate_scenes(...)` to get a list of `Scene` objects.
- Loops through each scene and calls `generate_video(...)`, collecting the resulting file paths.
- Calls `merge_videos(...)` to create a single `vlog.mp4` inside `output_dir`.

The `if __name__ == "__main__":` block in the script runs this pipeline with the Paris + kangaroo example.

---

## Running the Script

From the `GenAIVideoGenerationVeo3` folder:

```bash
python main.py
```

or, if you changed the file name:

```bash
python your_script_name.py
```

You should see logs such as:

- Scene generation prompt being sent to Gemini.
- “Generating video…” followed by periodic “Waiting for video to generate…” messages.
- MoviePy writing the merged `vlog.mp4`.

After it finishes, check:

```text
GenAIVideoGenerationVeo3/
  └─ paris_2/
       ├─ scenes.md
       ├─ video_0.mp4
       ├─ video_1.mp4
       ├─ video_2.mp4
       ├─ video_3.mp4
       └─ vlog.mp4
```

---

## Customization Ideas

- **Change the story**  
  Update `idea` and `character_description` to create completely different vlogs (travel, tutorials, mini-movies, product ads, etc.).

- **Adjust style and mood**  
  Experiment with:
  - `video_characteristics` (e.g. `moody, film grain, handheld, documentary style`)
  - `camera_angle` (e.g. `first-person POV`, `drone shots`, `static tripod`)

- **Scene pacing**  
  Change `number_of_scenes` to control video length and pacing.

- **Post-processing**  
  Extend the MoviePy pipeline with:
  - Intro/outro screens
  - Transitions
  - Overlaid titles or background tracks

This project is meant as a **playground** and a **portfolio piece** for GenAI video workflows powered by Gemini and Veo 3.
