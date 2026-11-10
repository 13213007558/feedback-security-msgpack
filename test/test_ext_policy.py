import datetime

import pytest

import msgpack
from msgpack import ExtType, Timestamp
from msgpack.fallback import EXT_ALLOW, EXT_KEEP, EXT_REJECT, MAP_KEY, Unpacker

pytestmark = pytest.mark.skipif(
    msgpack.Unpacker.__module__ != "msgpack.fallback",
    reason="ext_policy is only supported by the pure Python unpacker",
)


def demo_policy(code, data_len, path):
    """Allow timestamp only at audit.created_at; keep unknown exts as ExtType."""
    if code == -1:
        if path == ("audit", "created_at"):
            return EXT_ALLOW
        return EXT_REJECT
    return EXT_KEEP


def pack(obj, **kwargs):
    return msgpack.packb(obj, **kwargs)


def test_demo_policy_allows_timestamp_at_configured_field():
    packed = pack(
        {
            "audit": {"created_at": Timestamp(1700000000, 123000000)},
            "trace_id": ExtType(7, b"\x01\x02"),
        }
    )
    obj = msgpack.unpackb(packed, timestamp=3, ext_policy=demo_policy)
    assert obj["audit"]["created_at"] == datetime.datetime(
        2023, 11, 14, 22, 13, 20, 123000, tzinfo=datetime.timezone.utc
    )
    # Unknown extensions elsewhere are kept as ExtType, ext_hook not called.
    assert obj["trace_id"] == ExtType(7, b"\x01\x02")


def test_demo_policy_rejects_timestamp_elsewhere():
    packed = pack({"audit": {"created_at": Timestamp(1, 0)}, "updated_at": Timestamp(2, 0)})
    with pytest.raises(ValueError, match=r"ext type -1 rejected by ext_policy at \$\.updated_at"):
        msgpack.unpackb(packed, timestamp=3, ext_policy=demo_policy)


def test_reject_error_message_has_path_and_code_but_no_payload():
    secret = b"sensitive-payload"
    packed = pack({"field": ExtType(9, secret)})
    policy = lambda code, data_len, path: EXT_REJECT
    with pytest.raises(ValueError) as excinfo:
        msgpack.unpackb(packed, ext_policy=policy)
    message = str(excinfo.value)
    assert "$.field" in message
    assert "ext type 9" in message
    assert secret.decode() not in message


def test_policy_sees_code_length_and_path():
    calls = []
    packed = pack({"items": [ExtType(3, b"abcd"), {"inner": ExtType(4, b"xy")}]})

    def policy(code, data_len, path):
        calls.append((code, data_len, path))
        return EXT_KEEP

    obj = msgpack.unpackb(packed, ext_policy=policy)
    assert calls == [
        (3, 4, ("items", 0)),
        (4, 2, ("items", 1, "inner")),
    ]
    assert obj == {"items": [ExtType(3, b"abcd"), {"inner": ExtType(4, b"xy")}]}


def test_map_key_position_is_reported_before_ext_hook_runs():
    calls = []
    hook_calls = []
    packed = pack({ExtType(7, b"k"): 1})

    def policy(code, data_len, path):
        calls.append(path)
        return EXT_REJECT

    def ext_hook(code, data):
        hook_calls.append(code)
        return "decoded"

    with pytest.raises(ValueError, match=r"rejected by ext_policy at \$\[<map key>\]"):
        msgpack.unpackb(packed, ext_policy=policy, ext_hook=ext_hook)
    assert calls == [(MAP_KEY,)]
    # The key was rejected before the decode callback ran.
    assert hook_calls == []


def test_map_key_still_checked_against_strict_map_key():
    packed = pack({ExtType(7, b"k"): 1})
    policy = lambda code, data_len, path: EXT_KEEP
    with pytest.raises(ValueError, match="is not allowed for map key"):
        msgpack.unpackb(packed, ext_policy=policy)
    # An ext_hook producing str is still usable as a key.
    obj = msgpack.unpackb(
        packed,
        ext_policy=lambda code, data_len, path: EXT_ALLOW,
        ext_hook=lambda code, data: "key",
    )
    assert obj == {"key": 1}


def test_nested_array_indices_in_path():
    packed = pack([[ExtType(1, b"a")], [ExtType(2, b"b"), [ExtType(3, b"c")]]])
    calls = []

    def policy(code, data_len, path):
        calls.append(path)
        return EXT_KEEP

    msgpack.unpackb(packed, ext_policy=policy)
    assert calls == [(0, 0), (1, 0), (1, 1, 0)]


def test_path_uses_raw_keys_not_object_hook_renames():
    packed = pack({"audit": {"created_at": Timestamp(1, 0)}})
    calls = []

    def policy(code, data_len, path):
        calls.append(path)
        return demo_policy(code, data_len, path)

    def object_hook(obj):
        return {("renamed_" + k): v for k, v in obj.items()}

    obj = msgpack.unpackb(packed, ext_policy=policy, object_hook=object_hook)
    # The policy saw the raw wire keys, so the timestamp was allowed.
    assert calls == [("audit", "created_at")]
    assert obj["renamed_audit"]["renamed_created_at"] == Timestamp(1, 0)


def test_chunked_feeding_hits_same_policy_positions():
    packed = pack(
        {
            "audit": {"created_at": Timestamp(1, 0)},
            "tags": [ExtType(5, b"x"), ExtType(6, b"yy")],
        }
    )
    oneshot_calls = []

    def recording(calls):
        def policy(code, data_len, path):
            calls.append((code, path))
            return demo_policy(code, data_len, path)

        return policy

    msgpack.unpackb(packed, timestamp=3, ext_policy=recording(oneshot_calls))

    chunked_calls = []
    unpacker = Unpacker(timestamp=3, ext_policy=recording(chunked_calls))
    results = []
    for i in range(len(packed)):
        unpacker.feed(packed[i : i + 1])
        results.extend(unpacker)
    assert len(results) == 1

    # Retries after OutOfData may consult the policy again for the same
    # position; the distinct positions hit must match the one-shot order.
    def dedup(calls):
        out = []
        for call in calls:
            if call not in out:
                out.append(call)
        return out

    assert dedup(chunked_calls) == oneshot_calls
    assert unpacker._ext_path == []


def test_policy_state_is_per_unpacker():
    inner_paths = []

    def inner_policy(code, data_len, path):
        inner_paths.append(path)
        return EXT_KEEP

    def ext_hook(code, data):
        # A nested Unpacker created inside a callback has its own path state.
        return msgpack.unpackb(data, ext_policy=inner_policy)

    packed = pack({"outer": ExtType(1, pack({"deep": ExtType(2, b"z")}))})
    obj = msgpack.unpackb(packed, ext_policy=lambda c, n, p: EXT_ALLOW, ext_hook=ext_hook)
    assert inner_paths == [("deep",)]
    assert obj == {"outer": {"deep": ExtType(2, b"z")}}


def test_invalid_policy_decision():
    packed = pack([ExtType(1, b"a")])
    with pytest.raises(ValueError, match="invalid decision"):
        msgpack.unpackb(packed, ext_policy=lambda code, n, path: "maybe")


@pytest.mark.parametrize("ext_policy", [0, "policy", []])
def test_invalid_ext_policy_type(ext_policy):
    with pytest.raises(TypeError, match="ext_policy.*callable"):
        msgpack.Unpacker(ext_policy=ext_policy)


def test_default_behavior_unchanged_without_policy():
    packed = pack({"ts": Timestamp(1, 0), "ext": ExtType(2, b"ab")})
    obj = msgpack.unpackb(packed)
    assert obj == {"ts": Timestamp(1, 0), "ext": ExtType(2, b"ab")}
    obj = msgpack.unpackb(packed, ext_hook=lambda code, data: (code, data))
    assert obj == {"ts": Timestamp(1, 0), "ext": (2, b"ab")}
