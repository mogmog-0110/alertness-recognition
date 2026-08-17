"""映像入力源（FrameSource）の実装と、データセット非依存の再標本化部品。"""

from .frame_rate import DownsampledFrameSource, downsample_frames, validate_downsample_fps
from .video_file import VideoFileSource, probe_video_fps

__all__ = [
    "DownsampledFrameSource",
    "VideoFileSource",
    "downsample_frames",
    "probe_video_fps",
    "validate_downsample_fps",
]
