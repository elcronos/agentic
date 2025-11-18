import os
from moviepy import VideoFileClip, concatenate_videoclips

def merge_videos(
    video_files: list[str], output_file: str = "vlog.mp4", output_dir: str = "videos"
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


merge_videos(["video_0.mp4","video_1.mp4","video_2.mp4","video_3.mp4",], "vlog.mp4", output_dir=".")
