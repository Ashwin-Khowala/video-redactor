from video_redactor.config import MatchConfig, ProcessingConfig
from video_redactor.processor import resolve_output_path
import os
import unittest.mock as mock

def test_config_instantiation():
    match_cfg = MatchConfig(mode="patterns", custom_keywords=["secret"])
    config = ProcessingConfig(
        input_path="input.mp4",
        output_path="output.mp4",
        match_config=match_cfg,
        frame_skip=2,
        padding=15
    )
    
    assert config.input_path == "input.mp4"
    assert config.output_path == "output.mp4"
    assert config.match_config.mode == "patterns"
    assert "secret" in config.match_config.custom_keywords
    assert config.frame_skip == 2
    assert config.padding == 15
    assert config.threads == 4

def test_resolve_output_path():
    match_cfg = MatchConfig()
    
    config = ProcessingConfig(input_path="my_video.mp4", output_path="", match_config=match_cfg)
    resolve_output_path(config)
    assert config.output_path == "my_video_redacted.mp4"
    
    config = ProcessingConfig(input_path="sub/folder/video.avi", output_path="output_dir/", match_config=match_cfg)
    with mock.patch("os.makedirs") as mock_makedirs, mock.patch("os.path.isdir", return_value=True):
        resolve_output_path(config)
        assert config.output_path == os.path.join("output_dir/", "video_redacted.avi")
