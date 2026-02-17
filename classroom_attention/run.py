from app.pipeline.processor import ClassroomProcessor

if __name__ == "__main__":
    processor = ClassroomProcessor()

    processor.process_video(
        video_path="input.mp4",
        output_path="outputs/output.avi"
    )