"""Testing code for the tupa.oracle module, unit-testing only."""

import pytest

from tupa.action import Actions
from tupa.oracle import Oracle
from tupa.states.state import State
from .conftest import load_passages, Settings, passage_id


@pytest.mark.parametrize("setting", Settings.all(), ids=str)
@pytest.mark.parametrize("passage", load_passages(), ids=passage_id)
def test_oracle(config, setting, passage, write_oracle_actions):
    config.update(setting.dict())
    config.set_format(passage.extra.get("format") or "ucca")
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    actions_taken = []
    while True:
        action = min(oracle.get_actions(state, actions).values(), key=str)
        state.transition(action)
        s = str(action)
        if state.need_label:
            label, _ = oracle.get_label(state, action)
            state.label_node(label)
            s += " " + str(label)
        actions_taken.append(s + "\n")
        if state.finished:
            break
    compare_file = "test_files/oracle_actions/%s%s.txt" % (passage.ID, setting.suffix())
    if write_oracle_actions:
        with open(compare_file, "w") as f:
            f.writelines(actions_taken)
    with open(compare_file) as f:
        assert f.readlines() == actions_taken, compare_file
