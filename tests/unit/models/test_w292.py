"""Unit tests for wave-292 networking-4 canon modules."""

import hashlib

from quant_fund.models.doh_wire import decode_query, encode_name, encode_query
from quant_fund.models.qpack_pack import decode, encode
from quant_fund.models.quic_streams import reassemble
from quant_fund.models.sctp_tsn import cum_ack, deliver
from quant_fund.models.tls13_trans import derive, finished_mac, handshake_messages
from quant_fund.models.wg_ik import ik_handshake


def test_quic_reassemble():
    frames = [(1, 0, b"POST "), (0, 0, b"GET "), (0, 4, b"HTTP"), (0, 8, b"1.1"), (1, 5, b"/up")]
    assert reassemble(0, frames) == b"GET HTTP1.1"
    assert reassemble(1, frames) == b"POST /up"


def test_tls13_same_secret():
    ch, sh, fin = handshake_messages()
    shared = hashlib.sha256(b"s").digest()
    a, b = derive(ch, sh, fin, shared)
    c, d = derive(ch, sh, fin, shared)
    assert (a, b) == (c, d) and a != b


def test_tls13_finished():
    ch, sh, fin = handshake_messages()
    hs, _ap = derive(ch, sh, fin, hashlib.sha256(b"s").digest())
    m = finished_mac(hs[:32], ch + sh + fin)
    assert finished_mac(hs[:32], ch + sh + fin + b"x") != m


def test_qpack_static():
    block = encode([(":method", "GET"), (":status", "200")], {})
    assert decode(block, {}) == [(":method", "GET"), (":status", "200")]


def test_wg_ik_equal():
    ki, kr = ik_handshake()
    assert ki == kr and len(ki) == 32


def test_doh_roundtrip():
    wire = encode_query(7, "a.b.c")
    qid, name, _qt = decode_query(wire)
    assert qid == 7 and name == "a.b.c"
    assert encode_name("x") == b"\x01x\x00"


def test_sctp():
    chunks = [(10, 0, 0, b"A"), (11, 1, 0, b"x"), (12, 0, 1, b"B")]
    assert deliver(chunks)[0] == b"AB" and cum_ack(chunks) == 12
