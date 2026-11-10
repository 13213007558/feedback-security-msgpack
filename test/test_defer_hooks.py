#!/usr/bin/env python
"""Tests for Unpacker(defer_hooks=True): each complete value is constructed
exactly once, no matter how the input is fragmented."""

import io

from pytest import mark, raises

from msgpack import packb
from msgpack.exceptions import BufferFull, OutOfData
from msgpack.ext import ExtType
from msgpack.fallback import Unpacker, unpackb

MSG1 = {
    "list": [1, 2],
    "ext": ExtType(5, b"ab"),
    "map": {"x": []},
}
MSG2 = [ExtType(7, b"z"), {"k": "v"}]

# Expected callback log for MSG1 / MSG2 with the recording hooks below.
# Hooks transform values, so inner entries appear inside outer arguments,
# which also pins down the inner-to-outer call order.
EXPECTED_LOG_1 = [
    ("list_hook", [1, 2]),
    ("ext_hook", 5, b"ab"),
    ("list_hook", []),
    ("object_hook", {"x": ("list", [])}),
    (
        "object_hook",
        {
            "list": ("list", [1, 2]),
            "ext": ("ext", 5, b"ab"),
            "map": ("obj", {"x": ("list", [])}),
        },
    ),
]
EXPECTED_LOG_2 = [
    ("ext_hook", 7, b"z"),
    ("object_hook", {"k": "v"}),
    ("list_hook", [("ext", 7, b"z"), ("obj", {"k": "v"})]),
]
RESULT_1 = (
    "obj",
    {
        "list": ("list", [1, 2]),
        "ext": ("ext", 5, b"ab"),
        "map": ("obj", {"x": ("list", [])}),
    },
)
RESULT_2 = ("list", [("ext", 7, b"z"), ("obj", {"k": "v"})])

DATA = packb(MSG1) + packb(MSG2)
LEN_1 = len(packb(MSG1))


class HookLog:
    """Hooks that record every call and return a tagged value."""

    def __init__(self):
        self.calls = []

    def object_hook(self, d):
        self.calls.append(("object_hook", d))
        return ("obj", d)

    def list_hook(self, lst):
        self.calls.append(("list_hook", list(lst)))
        return ("list", list(lst))

    def ext_hook(self, code, data):
        self.calls.append(("ext_hook", code, data))
        return ("ext", code, data)

    def unpacker(self, **kwargs):
        return Unpacker(
            object_hook=self.object_hook,
            list_hook=self.list_hook,
            ext_hook=self.ext_hook,
            defer_hooks=True,
            **kwargs,
        )


def run_with_cuts(data, cuts):
    """Feed *data* split at *cuts*, attempting unpack() after every feed."""
    log = HookLog()
    unpacker = log.unpacker()
    results = []
    prev = 0
    for cut in list(cuts) + [len(data)]:
        unpacker.feed(data[prev:cut])
        prev = cut
        while True:
            try:
                results.append(unpacker.unpack())
            except OutOfData:
                break
    return results, log.calls


def test_complete_value_constructed_once():
    results, calls = run_with_cuts(DATA, [])
    assert results == [RESULT_1, RESULT_2]
    assert calls == EXPECTED_LOG_1 + EXPECTED_LOG_2


@mark.parametrize("cut", range(len(DATA) + 1))
def test_every_single_cut_point(cut):
    results, calls = run_with_cuts(DATA, [cut])
    assert results == [RESULT_1, RESULT_2]
    assert calls == EXPECTED_LOG_1 + EXPECTED_LOG_2


def test_every_pair_of_cut_points():
    for cut1 in range(len(DATA) + 1):
        for cut2 in range(cut1, len(DATA) + 1):
            results, calls = run_with_cuts(DATA, [cut1, cut2])
            assert results == [RESULT_1, RESULT_2]
            assert calls == EXPECTED_LOG_1 + EXPECTED_LOG_2


def test_byte_by_byte_feed():
    results, calls = run_with_cuts(DATA, range(len(DATA)))
    assert results == [RESULT_1, RESULT_2]
    assert calls == EXPECTED_LOG_1 + EXPECTED_LOG_2


def test_outofdata_retries_run_no_hooks():
    log = HookLog()
    unpacker = log.unpacker()
    unpacker.feed(DATA[: LEN_1 - 1])
    for _ in range(3):
        with raises(OutOfData):
            unpacker.unpack()
    assert log.calls == []
    assert unpacker.tell() == 0
    unpacker.feed(DATA[LEN_1 - 1 :])
    assert unpacker.unpack() == RESULT_1
    assert unpacker.tell() == LEN_1
    assert unpacker.unpack() == RESULT_2
    assert unpacker.tell() == len(DATA)
    assert log.calls == EXPECTED_LOG_1 + EXPECTED_LOG_2


def test_inner_success_outer_failure_consumes_message():
    log = HookLog()

    def object_hook(d):
        log.calls.append(("object_hook", d))
        if "list" in d:
            raise RuntimeError("outer hook failed")
        return ("obj", d)

    unpacker = Unpacker(
        object_hook=object_hook,
        list_hook=log.list_hook,
        ext_hook=log.ext_hook,
        defer_hooks=True,
    )
    unpacker.feed(DATA)
    with raises(RuntimeError, match="outer hook failed"):
        unpacker.unpack()
    # Inner hooks ran exactly once; the outer hook failed exactly once.
    assert log.calls == EXPECTED_LOG_1
    # The failed message is consumed: tell() sits at its end and the next
    # unpack continues with the second message instead of re-running hooks.
    assert unpacker.tell() == LEN_1
    assert unpacker.unpack() == RESULT_2
    assert unpacker.tell() == len(DATA)
    assert log.calls == EXPECTED_LOG_1 + EXPECTED_LOG_2


def test_ext_hook_failure_consumes_message():
    log = HookLog()

    def ext_hook(code, data):
        log.calls.append(("ext_hook", code, data))
        if code == 5:
            raise RuntimeError("ext hook failed")
        return ("ext", code, data)

    unpacker = Unpacker(
        object_hook=log.object_hook,
        list_hook=log.list_hook,
        ext_hook=ext_hook,
        defer_hooks=True,
    )
    unpacker.feed(DATA)
    with raises(RuntimeError, match="ext hook failed"):
        unpacker.unpack()
    # list_hook ran once, then ext_hook failed once; nothing is retried.
    assert log.calls == [("list_hook", [1, 2]), ("ext_hook", 5, b"ab")]
    assert unpacker.tell() == LEN_1
    assert unpacker.unpack() == RESULT_2
    assert log.calls == [
        ("list_hook", [1, 2]),
        ("ext_hook", 5, b"ab"),
        ("ext_hook", 7, b"z"),
        ("object_hook", {"k": "v"}),
        ("list_hook", [("ext", 7, b"z"), ("obj", {"k": "v"})]),
    ]


def test_object_pairs_hook_return_value_and_order():
    calls = []

    def object_pairs_hook(pairs):
        calls.append(("object_pairs_hook", list(pairs)))
        return ("pairs", list(pairs))

    unpacker = Unpacker(
        object_pairs_hook=object_pairs_hook,
        list_hook=HookLog().list_hook,
        defer_hooks=True,
    )
    unpacker.feed(packb({"a": [1], "b": {"c": 2}}))
    assert unpacker.unpack() == (
        "pairs",
        [("a", ("list", [1])), ("b", ("pairs", [("c", 2)]))],
    )
    assert calls == [
        ("object_pairs_hook", [("c", 2)]),
        ("object_pairs_hook", [("a", ("list", [1])), ("b", ("pairs", [("c", 2)]))]),
    ]


def test_hook_option_validation_unchanged():
    with raises(TypeError, match="mutually exclusive"):
        Unpacker(object_hook=dict, object_pairs_hook=list, defer_hooks=True)
    with raises(TypeError, match="not callable"):
        Unpacker(list_hook=42, defer_hooks=True)
    with raises(TypeError, match="not callable"):
        Unpacker(ext_hook=None, defer_hooks=True)


def test_use_list_false_applies_after_list_hook():
    log = HookLog()
    unpacker = log.unpacker(use_list=False)
    unpacker.feed(packb([[1], 2]))
    assert unpacker.unpack() == ("list", [("list", [1]), 2])
    assert log.calls == [("list_hook", [1]), ("list_hook", [("list", [1]), 2])]


def test_no_hooks_matches_plain_unpack():
    obj = {"a": [1, "x", b"y", None], "b": {"c": [True, 1.5]}}
    assert unpackb(packb(obj), defer_hooks=True) == obj
    assert unpackb(packb(obj), defer_hooks=True, use_list=False) == {
        "a": (1, "x", b"y", None),
        "b": {"c": (True, 1.5)},
    }


def test_limits_still_enforced():
    log = HookLog()
    with raises(ValueError, match="exceeds max_array_len"):
        unpackb(
            packb([1, 2, 3]),
            max_array_len=2,
            defer_hooks=True,
            list_hook=log.list_hook,
        )
    assert log.calls == []
    with raises(ValueError, match="exceeds max_str_len"):
        unpackb(packb({"abc": 1}), max_str_len=2, defer_hooks=True)
    with raises(ValueError, match="exceeds max_ext_len"):
        unpackb(
            packb(ExtType(1, b"abcd")),
            max_ext_len=2,
            defer_hooks=True,
            ext_hook=log.ext_hook,
        )
    assert log.calls == []
    unpacker = log.unpacker(max_buffer_size=4)
    with raises(BufferFull):
        unpacker.feed(packb(MSG1))


def test_strict_map_key_checked_on_constructed_keys():
    with raises(ValueError, match="is not allowed for map key"):
        unpackb(packb({1: 2}), defer_hooks=True)
    assert unpackb(packb({1: 2}), strict_map_key=False, defer_hooks=True) == {1: 2}


def test_file_like_iteration():
    log = HookLog()
    unpacker = log.unpacker(file_like=io.BytesIO(DATA))
    assert list(unpacker) == [RESULT_1, RESULT_2]
    assert log.calls == EXPECTED_LOG_1 + EXPECTED_LOG_2


def test_legacy_mode_still_repeats_hooks():
    # Documents why defer_hooks exists: without it, an OutOfData retry
    # re-runs hooks for inner values that were already complete.
    log = HookLog()
    unpacker = Unpacker(
        object_hook=log.object_hook,
        list_hook=log.list_hook,
        ext_hook=log.ext_hook,
    )
    cut = 1 + 5 + 3  # map header + "list" + [1, 2]
    unpacker.feed(DATA[:cut])
    with raises(OutOfData):
        unpacker.unpack()
    unpacker.feed(DATA[cut:])
    assert unpacker.unpack() == RESULT_1
    assert unpacker.unpack() == RESULT_2
    # The inner [1, 2] list_hook fired once per attempt.
    assert log.calls.count(("list_hook", [1, 2])) == 2
