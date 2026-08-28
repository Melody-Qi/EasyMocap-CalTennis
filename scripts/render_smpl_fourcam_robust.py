"""Render multi-camera SMPL overlays while preserving frames with no people."""

import argparse
import os
from types import SimpleNamespace

from tqdm import tqdm

from easymocap.config.baseconfig import load_object_from_cmd
from easymocap.mytools.vis_base import merge
from easymocap.visualize.ffmpeg_wrapper import VideoMaker
from easymocap.visualize.render_base import Images, MIOutputs, Results, imwrite


class RobustMIOutputs(MIOutputs):
    """EasyMocap's MIOutputs crashes on an empty per-frame result."""

    def __call__(self, images, results, cameras, basename):
        if len(results) == 0:
            output = merge(list(images.values()), square=True)
            imwrite(os.path.join(self.out, basename + self.ext), output)
            return 0
        return super().__call__(images, results, cameras, basename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subs", nargs="+", required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    body_model = load_object_from_cmd(args.model, [])
    results = Results(body_model=body_model, path=args.result, rend_type="mesh")
    image_args = SimpleNamespace(
        scale=1.0, undis=True, tgt_shape=[-1, -1], ext=".jpg"
    )
    inputs = Images(path=args.path, subs=args.subs, image_args=image_args)
    outputs = RobustMIOutputs(
        out=args.output, mode="image", backend="pyrender", merge=True
    )

    for index in tqdm(range(len(results)), desc="render-smpl-fourcam"):
        basename, result = results[index]
        images, cameras = inputs(basename)
        outputs(images, result, cameras, basename)

    VideoMaker(
        restart=True,
        fps_in=args.fps,
        fps_out=args.fps,
        remove_images=False,
    ).make_video(args.output)


if __name__ == "__main__":
    main()

