"""Testing code for the tupa.oracle module, unit-testing only."""

import pytest

from tupa.action import Actions
from tupa.oracle import Oracle
from tupa.states.state import State
from .conftest import passage_files, Settings, load_passage, basename


@pytest.mark.parametrize("setting", Settings.all(), ids=str)
@pytest.mark.parametrize("filename", passage_files(), ids=basename)
def test_oracle(config, setting, filename, write_oracle_actions):
    passage = load_passage(filename)
    config.update(setting.dict())
    config.set_format(passage.extra.get("format") or "ucca")
    compare_file = "test_files/oracle_actions/%s%s.txt" % (basename(filename), setting.suffix())
    actions_taken = []
    with open(compare_file, "rw"[write_oracle_actions], encoding="utf-8") as f:
        for i, action in enumerate(gen_actions(passage)):
            if write_oracle_actions:
                print(action, file=f)
            else:
                assert action == f.readline().strip(), "Action %d does not match expected (all actions taken: %s)" % (
                    i, ", ".join(actions_taken) or "none")
            actions_taken.append(action)
    assert len(actions_taken) > 2, passage


def gen_actions(passage):
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    while True:
        action = min(oracle.get_actions(state, actions).values(), key=str)
        state.transition(action)
        s = str(action)
        if state.need_label:
            label, _ = oracle.get_label(state, action)
            state.label_node(label)
            s += " " + str(label)
        yield s
        if state.finished:
            break
