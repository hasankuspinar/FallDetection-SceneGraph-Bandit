import argparse
from fall_graph.config import load_config
from fall_graph.video import process_video, summarize_results, summarize_fall_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--fps_target", type=int, default=3, help="Sampling FPS target")
    ap.add_argument("--logs_dir", default="video_logs", help="Output directory")
    ap.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    args = ap.parse_args()

    cfg = load_config(args.config)

    results, fps, total_frames = process_video(
        video_path=args.video,
        fps_target=args.fps_target,
        logs_dir=args.logs_dir,
        cfg=cfg,
    )

    summarize_results(results, total_frames=total_frames, fps_target=args.fps_target)
    summarize_fall_score(results, total_frames=total_frames, fps_target=fps)

if __name__ == "__main__":
    main()
