import cv2
import os
from pathlib import Path

# Loop through s1 to s5
for speaker_num in range(1, 6):
    speaker_id = f"s{speaker_num}"
    input_dir = Path(f"../../grid_data/{speaker_id}/{speaker_id}")
    output_dir = Path(f"../../grid_preprocessed/{speaker_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Processing {speaker_id}...")

    def extract_frames(video_path, output_folder):
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = output_folder / f"{video_path.stem}_frame{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_idx += 1
        cap.release()

    for file in input_dir.glob("*.mpg"):
        file_out_dir = output_dir / file.stem
        file_out_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(file, file_out_dir)

    print(f"✅ Done: {speaker_id}")

print("🎉 Frame extraction complete for s1 to s5.")

