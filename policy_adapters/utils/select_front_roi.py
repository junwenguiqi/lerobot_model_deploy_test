from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parents[2]
    / "dataset_tactile"
    / "fr3_zed_raw_tactile_tashan_train56"
)
DEFAULT_SAVE_PATH = Path(__file__).resolve().parents[2] / "outputs" / "front_roi_preview.png"


Roi = tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select or verify a fixed front-camera ROI."
    )
    parser.add_argument("--image", type=Path, default=None, help="Direct front image path.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--roi", default=None, help="Existing ROI: x1,y1,x2,y2.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--no-show", action="store_true", help="Requires --roi.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args)
    image = load_rgb(image_path)

    roi = parse_roi(args.roi, image.size) if args.roi else None
    if roi is None:
        if args.no_show:
            raise ValueError("--no-show requires --roi")
        roi = select_roi_interactive(image, image_path)

    save_path = args.save.resolve()
    save_preview(image, crop_resize(image, roi, args.size), roi, save_path, args.size)
    print_result(roi, args.size, save_path)


def resolve_image_path(args: argparse.Namespace) -> Path:
    if args.image is not None:
        return args.image.expanduser().resolve()
    return front_frame_path(args.dataset_root, args.episode, args.frame)


def front_frame_path(dataset_root: Path, episode: int, frame: int) -> Path:
    return (
        dataset_root.expanduser().resolve()
        / "image"
        / "front"
        / f"episode_{int(episode):06d}"
        / f"frame_{int(frame):06d}.png"
    )


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def parse_roi(value: str, image_size: tuple[int, int]) -> Roi:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must be x1,y1,x2,y2, got {value!r}")
    try:
        roi = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"ROI values must be integers, got {value!r}") from exc
    return validate_roi(roi, image_size)


def validate_roi(roi: tuple[int, ...], image_size: tuple[int, int]) -> Roi:
    left, top, right, bottom = (int(x) for x in roi)
    width, height = image_size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"ROI {(left, top, right, bottom)} is outside image size {image_size}")
    if right <= left or bottom <= top:
        raise ValueError(f"ROI must satisfy x2>x1 and y2>y1, got {(left, top, right, bottom)}")
    return left, top, right, bottom


def select_roi_interactive(image: Image.Image, image_path: Path) -> Roi:
    selected: dict[str, Roi] = {}
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.imshow(image)
    ax.set_title(f"Drag front ROI, then close window\n{image_path}", fontsize=11)
    ax.axis("off")

    def on_select(eclick: object, erelease: object) -> None:
        points = (
            getattr(eclick, "xdata", None),
            getattr(eclick, "ydata", None),
            getattr(erelease, "xdata", None),
            getattr(erelease, "ydata", None),
        )
        if any(point is None for point in points):
            return
        selected["roi"] = roi_from_drag(*(float(point) for point in points), image.size)
        print(f"selected ROI: {format_roi(selected['roi'])}")

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=False,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props={"facecolor": "none", "edgecolor": "red", "linewidth": 2},
        handle_props={"markeredgecolor": "red"},
    )
    fig._front_roi_selector = selector
    plt.show()
    if "roi" not in selected:
        raise RuntimeError("No ROI selected.")
    return selected["roi"]


def roi_from_drag(x0: float, y0: float, x1: float, y1: float, image_size: tuple[int, int]) -> Roi:
    width, height = image_size
    roi = (
        max(0, min(width - 1, math.floor(min(x0, x1)))),
        max(0, min(height - 1, math.floor(min(y0, y1)))),
        max(1, min(width, math.ceil(max(x0, x1)))),
        max(1, min(height, math.ceil(max(y0, y1)))),
    )
    return validate_roi(roi, image_size)


def crop_resize(image: Image.Image, roi: Roi, size: int) -> Image.Image:
    if size <= 0:
        raise ValueError(f"--size must be positive, got {size}")
    return image.crop(roi).resize((int(size), int(size)), Image.Resampling.BILINEAR)


def save_preview(
    image: Image.Image,
    resized: Image.Image,
    roi: Roi,
    save_path: Path,
    size: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    left, top, right, bottom = roi

    axes[0].imshow(image)
    axes[0].add_patch(
        Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="red", linewidth=2)
    )
    axes[0].set_title(f"front original\nROI {format_roi(roi)}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(resized)
    axes[1].set_title(f"ROI -> {int(size)}x{int(size)}", fontsize=10)
    axes[1].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def print_result(roi: Roi, size: int, save_path: Path) -> None:
    roi_text = format_roi(roi)
    left, top, right, bottom = roi
    print(f"Saved preview: {save_path}")
    print(f'FRONT_ROI="{roi_text}"')
    print(f"FRONT_ROI_SIZE={int(size)}")
    print(f'--front-roi "{roi_text}" --front-roi-size {int(size)}')
    print(f"width={right - left}, height={bottom - top}")


def format_roi(roi: Roi) -> str:
    return ",".join(str(int(x)) for x in roi)


if __name__ == "__main__":
    main()
