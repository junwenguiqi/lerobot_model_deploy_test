from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .select_front_roi import (
        DEFAULT_DATASET_ROOT,
        crop_resize,
        front_frame_path,
        load_rgb,
        parse_roi,
        print_result,
        save_preview,
    )
except ImportError:
    from select_front_roi import (  # type: ignore
        DEFAULT_DATASET_ROOT,
        crop_resize,
        front_frame_path,
        load_rgb,
        parse_roi,
        print_result,
        save_preview,
    )


DEFAULT_SAVE_PATH = Path(__file__).resolve().parents[2] / "outputs" / "front_roi_resize_256.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview front ROI resize to policy input size.")
    parser.add_argument("--image", type=Path, default=None, help="Direct front image path.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--roi", default=None, help="Front ROI: x1,y1,x2,y2.")
    parser.add_argument("--front-crop-box", default=None, help="Alias of --roi.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--no-show", action="store_true", help="Kept for old commands; preview is always saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roi_text = args.roi or args.front_crop_box
    if roi_text is None:
        raise ValueError("Please pass --roi or --front-crop-box, for example: --roi 318,104,736,540")

    image_path = args.image.expanduser().resolve() if args.image else front_frame_path(
        args.dataset_root,
        args.episode,
        args.frame,
    )
    image = load_rgb(image_path)
    roi = parse_roi(roi_text, image.size)
    save_path = args.save.resolve()

    save_preview(image, crop_resize(image, roi, args.size), roi, save_path, args.size)
    print_result(roi, args.size, save_path)


if __name__ == "__main__":
    main()
