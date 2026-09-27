from __future__ import annotations

import pytest

from vos_memory_inspector.replay import plan_original_prompt_replay


def test_original_prompt_replay_plan_uses_anchor_and_exact_recent_window() -> None:
    plan = plan_original_prompt_replay(
        switch_frame=167,
        replay_frames=8,
        original_prompt_frame=0,
    )

    assert plan["method"] == "original_prompt_replay_8"
    assert plan["original_prompt_frame"] == 0
    assert plan["replay_window"] == list(range(160, 168))
    assert plan["unique_past_rgb_frame_count"] == 9
    assert plan["unique_past_rgb_frames"] == [0, *range(160, 168)]
    assert plan["uses_ground_truth_original_prompt"]
    assert not plan["uses_source_prediction_prompt"]


@pytest.mark.parametrize("replay_frames", [4, 8, 16])
def test_protocol_replay_lengths_end_at_switch(replay_frames: int) -> None:
    plan = plan_original_prompt_replay(
        switch_frame=40,
        replay_frames=replay_frames,
    )
    assert len(plan["replay_window"]) == replay_frames
    assert plan["replay_window"][-1] == 40
    assert plan["replay_start_frame"] == 40 - replay_frames + 1


def test_original_prompt_replay_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        plan_original_prompt_replay(switch_frame=10, replay_frames=0)
    with pytest.raises(ValueError, match="available prefix"):
        plan_original_prompt_replay(switch_frame=3, replay_frames=5)
    with pytest.raises(ValueError, match="before the recent replay window"):
        plan_original_prompt_replay(
            switch_frame=10,
            replay_frames=4,
            original_prompt_frame=8,
        )
