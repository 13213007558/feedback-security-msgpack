#!/usr/bin/env python
"""Tests for the strict skip mode of the pure-Python Unpacker."""

import datetime

from pytest import mark, raises

from msgpack import ExtType, FormatError, OutOfData, StackError, packb
from msgpack.fallback import Unpacker


def bomb(*args):
    raise AssertionError("hook must not be called during skip")


def test_skip_calls_no_hooks():
    # An object exercising every constructor path: map, array, ext and
    # timestamp.  All hooks are traps; skip() must not call any of them.
    obj = {
        "list": [1, {"nested": "map"}, ExtType(2, b"zz")],
        "ext": ExtType(1, b"payload"),
        "ts": datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    }
    packed = packb(obj, datetime=True)
    unpacker = Unpacker(ext_hook=bomb, object_hook=bomb, list_hook=bomb, timestamp=3)
    unpacker.feed(packed)
    assert unpacker.skip() is None
    assert unpacker.tell() == len(packed)


def test_skip_calls_no_hooks_pairs():
    packed = packb({"a": [1, 2], "b": ExtType(3, b"x")})
    unpacker = Unpacker(object_pairs_hook=bomb, list_hook=bomb, ext_hook=bomb)
    unpacker.feed(packed)
    unpacker.skip()
    assert unpacker.tell() == len(packed)


# Same short byte strings, unpacked strictly and skipped: both paths must
# agree on accept/reject.
CASES = [
    # (packed, kwargs, expected exception type or None)
    (b"\xc0", {}, None),
    (packb({"a": [1, 2.5, b"x", "y"]}), {}, None),
    (packb(ExtType(1, b"ab")), {}, None),
    (packb("héllo"), {}, None),
    (b"\xc1", {}, FormatError),  # undefined tag
    (b"\x91\xc1", {}, FormatError),  # undefined tag inside array
    (b"\xa1\xff", {}, UnicodeDecodeError),  # invalid UTF-8 in raw
    (b"\xa1\xff", {"raw": True}, None),  # bytes: no UTF-8 constraint
    (b"\x81\x01\x02", {}, ValueError),  # int map key, strict_map_key
    (b"\x81\x91\x01\x02", {}, ValueError),  # array map key
    (b"\x81\x01\x02", {"strict_map_key": False}, None),
    (b"\x91" * 2000, {}, StackError),  # too deep
    (packb("abcd"), {"max_str_len": 3}, ValueError),
    (packb(b"abcd"), {"max_bin_len": 3}, ValueError),
    (packb([1, 2, 3]), {"max_array_len": 2}, ValueError),
    (packb({"a": 1, "b": 2}), {"max_map_len": 1}, ValueError),
    (packb(ExtType(1, b"abcd")), {"max_ext_len": 3}, ValueError),
    # ext32 with huge length followed by what looks like another object:
    # the trailing bytes must not be mistaken for the next message.
    (b"\xc9\xff\xff\xff\xff" + packb("next"), {}, ValueError),
]


@mark.parametrize("packed,kwargs,expected", CASES)
def test_skip_matches_strict_unpack(packed, kwargs, expected):
    for method in ("unpack", "skip"):
        unpacker = Unpacker(**kwargs)
        unpacker.feed(packed)
        if expected is None:
            getattr(unpacker, method)()
            assert unpacker.tell() == len(packed)
        else:
            with raises(expected):
                getattr(unpacker, method)()


def test_skip_middle_of_three_messages():
    messages = [packb(m) for m in ([1, 2, 3], {"mid": "value"}, "tail")]
    unpacker = Unpacker()
    unpacker.feed(b"".join(messages))
    unpacker.skip()
    assert unpacker.unpack() == {"mid": "value"}
    unpacker.skip()
    assert unpacker.tell() == sum(len(m) for m in messages)
    with raises(OutOfData):
        unpacker.unpack()


def test_malformed_message_does_not_skip_to_third():
    # Three concatenated messages; the first one is malformed (int map key
    # under strict_map_key).  skip() must fail at the malformed message
    # instead of silently landing on the well-formed third message.
    bad = b"\x81\x01\x02"  # {1: 2} -- int key rejected by strict_map_key
    third = packb("third")
    unpacker = Unpacker()
    unpacker.feed(bad + packb("second") + third)
    with raises(ValueError):
        unpacker.skip()
    # Nothing was confirmed: the stream position stays at the broken
    # message, and the error is not swallowed.
    assert unpacker.tell() == 0


def test_malformed_nested_message_fails_in_place():
    # Undefined tag inside the first message, followed by a valid message.
    unpacker = Unpacker()
    unpacker.feed(b"\x91\xc1" + packb("third"))
    with raises(FormatError):
        unpacker.skip()
    assert unpacker.tell() == 0


def test_skip_resumable_across_partial_feeds():
    packed = packb({"key": [1, 2, "three"], "ext": ExtType(9, b"xyz")})
    unpacker = Unpacker()
    # Feed a prefix that ends in the middle of the message.
    unpacker.feed(packed[:5])
    with raises(OutOfData):
        unpacker.skip()
    # OutOfData confirms nothing; feeding the rest lets skip() succeed
    # without consuming any header twice.
    assert unpacker.tell() == 0
    unpacker.feed(packed[5:])
    unpacker.skip()
    assert unpacker.tell() == len(packed)
    with raises(OutOfData):
        unpacker.skip()


def test_read_array_header_then_skip():
    packed = packb(["skip-me", "keep", 99])
    unpacker = Unpacker()
    unpacker.feed(packed)
    assert unpacker.read_array_header() == 3
    unpacker.skip()
    assert unpacker.unpack() == "keep"
    unpacker.skip()
    # tell() counts every consumed byte, headers and skipped items included.
    assert unpacker.tell() == len(packed)


def test_read_map_header_then_skip():
    packed = packb({"a": 1, "b": 2})
    unpacker = Unpacker()
    unpacker.feed(packed)
    assert unpacker.read_map_header() == 2
    unpacker.skip()  # key "a"
    unpacker.skip()  # value 1
    assert unpacker.unpack() == "b"
    assert unpacker.unpack() == 2
    assert unpacker.tell() == len(packed)


def test_limits_apply_after_read_array_header():
    # Limits are cumulative: elements read after read_array_header() are
    # validated against the same max_*_len constraints.
    packed = packb(["ok", "toolong"])
    unpacker = Unpacker(max_str_len=3)
    unpacker.feed(packed)
    assert unpacker.read_array_header() == 2
    unpacker.skip()
    with raises(ValueError):
        unpacker.skip()


def test_deep_nesting_rejected_before_expansion():
    # A deeply nested prefix with no elements at all: the depth limit must
    # fire from the headers alone, before any OutOfData.
    unpacker = Unpacker()
    unpacker.feed(b"\x91" * 2000)
    with raises(StackError):
        unpacker.skip()
