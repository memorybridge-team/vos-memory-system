"""기억을 넘기지 않는 두 기준.

Source-only : Base+ 로 넘기지 않고 Small이 끝까지 간다. Base+ 세션을 쓰지 않음.
Full Replay : Base+ 가 처음 프레임부터 전부 다시 본다. 회복률의 기준(100점).

둘 다 전환 시점과 상관없이 전환 뒤 결과가 같다 → evaluate_video.py 가 영상·객체마다 한 번만 돌리고
전환 시점마다 잘라 쓴다. 그래서 다른 비교군과 달리 prepare 함수 대신 아래 함수만 있다.
"""


def full_replay(session, prompt_frame: int, prompt_mask) -> int:
    """Base+ 에 처음 프롬프트만 주고, 처음부터 추적하게 한다. 추적 시작 프레임을 돌려준다."""
    session.add_prompt(prompt_frame, prompt_mask)
    return prompt_frame
