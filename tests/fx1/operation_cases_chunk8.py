# Hand-checked FX-1 cases. Expectation dicts match Output.model_dump(mode="json").
# Datetimes are UTC instants serialized as "...Z". No fx1 import.
#
# Every binary fixture below is assembled byte-by-byte from its published format
# specification (Avro 1.12 object container, BSON 1.1) and every asserted number
# is derived from that construction with hashlib / len / struct, not from a prior
# run of the operation under test. All fixtures are SYNTHETIC: they are invented
# correctness inputs, never market data and never copied from receipts/.

CASES = {
    "plugins.read_arff": {
        "arguments": {"path": "tiny.arff", "offset": 0, "limit": 50},
        # 8 physical lines, 120 UTF-8 bytes; line byte starts are the running sum
        # of len(line) + 1 for the LF: 0, 15, 36, 59, 79, 85, 94, 111.
        "files": {
            "tiny.arff": (
                "@relation tiny\n"
                "@attribute n numeric\n"
                "@attribute c {up,down}\n"
                "@attribute s string\n"
                "@data\n"
                "1,up,abc\n"
                "{0 2.5,2 quoted}\n"
                '?,?,"q?"\n'
            )
        },
        "expect": {
            "relation": "tiny",
            "attributes": [
                {
                    "name": "n",
                    "kind": "numeric",
                    "declared_type": "numeric",
                    "levels": [],
                    "missing_count": 1,
                    "implicit_default_count": 0,
                },
                {
                    "name": "c",
                    "kind": "nominal",
                    "declared_type": "nominal",
                    "levels": ["up", "down"],
                    "missing_count": 1,
                    "implicit_default_count": 1,
                },
                {
                    "name": "s",
                    "kind": "string",
                    "declared_type": "string",
                    "levels": [],
                    "missing_count": 0,
                    "implicit_default_count": 0,
                },
            ],
            "rows": [
                {
                    "row_index": 0,
                    "source_line": 6,
                    "source_byte_start": 85,
                    "source_record_bytes": 8,
                    "representation": "dense",
                    "cells": ["1", "up", "abc"],
                    "implicit_default_columns": [],
                },
                {
                    "row_index": 1,
                    "source_line": 7,
                    "source_byte_start": 94,
                    "source_record_bytes": 16,
                    "representation": "sparse",
                    "cells": ["2.5", "up", "quoted"],
                    "implicit_default_columns": [1],
                },
                {
                    "row_index": 2,
                    "source_line": 8,
                    "source_byte_start": 111,
                    "source_record_bytes": 8,
                    "representation": "dense",
                    "cells": [None, None, "q?"],
                    "implicit_default_columns": [],
                },
            ],
            "row_count": 3,
            "dense_row_count": 2,
            "sparse_row_count": 1,
            "has_data_rows": True,
            "physical_line_count": 8,
            "numeric_encoding": "original_decimal_lexeme_binary64_range",
            "offset": 0,
            "has_more": False,
            "full_source_validated": True,
            "source_bytes": 120,
            "source_sha256": "e2994e9e3aa8fa25e3f5a15f060f0759aa389b78118ba664ddfc15a662b248dd",
        },
        "approx": False,
        "note": (
            "SYNTHETIC 8-line ARFF. Line lengths 14,20,22,19,5,8,16,8 each plus one LF "
            "sum to 120 bytes and give record starts 85/94/111; sha256 is over those 120 "
            "bytes. Sparse row {0 2.5,2 quoted} omits nominal column 1, so its cell is the "
            "first declared level 'up' and implicit_default_columns is [1]. Row 3 carries "
            'two unquoted ? (missing) and a quoted "q?" that stays a string literal, so '
            "n and c each gain one missing_count. Two dense rows, one sparse row."
        ),
    },
    "plugins.read_avro_container": {
        "arguments": {"path": "tiny.avro", "offset": 0, "limit": 50},
        # Obj\x01 | metadata map (187 B) | 16 B sync | one null-codec block:
        # zigzag(1)=0x02 count, zigzag(13)=0x1a size, 13 record bytes, 16 B sync.
        "files": {
            "tiny.avro": {
                "base64": (
                    "T2JqAQQWYXZyby5zY2hlbWG2AnsidHlwZSI6InJlY29yZCIsIm5hbWUiOiJSZWMiLCJmaWVsZHMi"
                    "Olt7Im5hbWUiOiJpIiwidHlwZSI6ImludCJ9LHsibmFtZSI6ImQiLCJ0eXBlIjoiZG91YmxlIn0s"
                    "eyJuYW1lIjoicyIsInR5cGUiOiJzdHJpbmcifSx7Im5hbWUiOiJmIiwidHlwZSI6ImJvb2xlYW4i"
                    "fV19FGF2cm8uY29kZWMIbnVsbAAAAQIDBAUGBwgJCgsMDQ4PAhoKAAAAAAAA+D8EaGkBAAECAwQF"
                    "BgcICQoLDA0ODw=="
                )
            }
        },
        "expect": {
            "record_fullname": "Rec",
            "columns": [
                {"name": "i", "primitive_type": "int"},
                {"name": "d", "primitive_type": "double"},
                {"name": "s", "primitive_type": "string"},
                {"name": "f", "primitive_type": "boolean"},
            ],
            "codec": "null",
            "metadata": [
                {
                    "key": "avro.schema",
                    "value_bytes": 155,
                    "value_sha256": "90e1f04b294b41a3a0da658b00f52189c07db44fb6f7c912c3211193940a77d3",
                },
                {
                    "key": "avro.codec",
                    "value_bytes": 4,
                    "value_sha256": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
                },
            ],
            "schema_sha256": "90e1f04b294b41a3a0da658b00f52189c07db44fb6f7c912c3211193940a77d3",
            "sync_marker_hex": "000102030405060708090a0b0c0d0e0f",
            "header_bytes": 207,
            "data_block_count": 1,
            "record_count": 1,
            "has_records": True,
            "expanded_record_bytes": 13,
            "rows": [
                {
                    "row_index": 0,
                    "block_index": 0,
                    "row_in_block": 0,
                    "block_source_byte_start": 209,
                    "block_source_byte_length": 13,
                    "expanded_byte_start": 0,
                    "expanded_byte_length": 13,
                    "source_byte_start": 209,
                    "cells": [
                        {"value": "5", "encoded_hex": "0a"},
                        {
                            "value": "0x1.8000000000000p+0",
                            "encoded_hex": "000000000000f83f",
                        },
                        {"value": "hi", "encoded_hex": "046869"},
                        {"value": True, "encoded_hex": "01"},
                    ],
                }
            ],
            "offset": 0,
            "has_more": False,
            "full_source_validated": True,
            "sync_marker_authenticates_source": False,
            "schema_resolution_performed": False,
            "source_bytes": 238,
            "source_sha256": "f7349089b7b0126800da937b9577918544a717597eafe09f838542dad03d2845",
        },
        "approx": False,
        "note": (
            "SYNTHETIC 238-byte OCF built with struct: 4-byte magic + 187-byte metadata map "
            "(zigzag count 2, 'avro.schema' -> 155-byte flat 4-field record, 'avro.codec' -> "
            "the 4 bytes 'null', zigzag 0 terminator) + sync bytes(range(16)); "
            "header_bytes = 4+187+16 = 207. "
            "The one null-codec block is zigzag(1)=0x02, zigzag(13)=0x1a, then 13 record bytes, "
            "so the block payload starts at 207+2 = 209 and is 13 bytes long. The record is "
            "zigzag(5)=0x0a, struct.pack('<d',1.5)=000000000000f83f whose float.hex is "
            "0x1.8000000000000p+0, length-2 string 'hi'=046869, boolean 0x01 -- 1+8+3+1 = 13 "
            "bytes, which is also expanded_record_bytes. Whole-file sha256 over those 238 bytes."
        ),
    },
    "plugins.read_bson": {
        "arguments": {"path": "tiny.bson", "offset": 0, "limit": 50},
        # <int32 108> then nine elements in order i,s,d,t,z,bin,re,sub,arr and a NUL.
        "files": {
            "tiny.bson": {
                "base64": (
                    "bAAAABBpAAcAAAACcwADAAAAYWIAAWQAAAAAAAAA+D8IdAABCnoABWJpbgACAAAAAN6tC3Jl"
                    "AGEuYwBpbQADc3ViABAAAAASagD7/////////wAEYXJyABMAAAAQMAABAAAAEDEAAgAAAAAA"
                )
            }
        },
        "expect": {
            "nodes": [
                {
                    "id": 0,
                    "parent_id": None,
                    "child_index": 0,
                    "depth": 0,
                    "name": None,
                    "byte_start": 0,
                    "byte_length": 108,
                    "value_byte_start": 0,
                    "value_byte_length": 108,
                    "type_byte_hex": None,
                    "kind": "document",
                    "value": None,
                    "child_count": 9,
                },
                {
                    "id": 1,
                    "parent_id": 0,
                    "child_index": 0,
                    "depth": 1,
                    "name": "i",
                    "byte_start": 4,
                    "byte_length": 7,
                    "value_byte_start": 7,
                    "value_byte_length": 4,
                    "type_byte_hex": "10",
                    "kind": "int32",
                    "value": "7",
                    "scalar_payload_hex": "07000000",
                },
                {
                    "id": 2,
                    "parent_id": 0,
                    "child_index": 1,
                    "depth": 1,
                    "name": "s",
                    "byte_start": 11,
                    "byte_length": 10,
                    "value_byte_start": 14,
                    "value_byte_length": 7,
                    "type_byte_hex": "02",
                    "kind": "string",
                    "value": "ab",
                    "scalar_payload_hex": None,
                },
                {
                    "id": 3,
                    "parent_id": 0,
                    "child_index": 2,
                    "depth": 1,
                    "name": "d",
                    "byte_start": 21,
                    "byte_length": 11,
                    "value_byte_start": 24,
                    "value_byte_length": 8,
                    "type_byte_hex": "01",
                    "kind": "double",
                    "value": "0x1.8000000000000p+0",
                    "scalar_payload_hex": "000000000000f83f",
                },
                {
                    "id": 4,
                    "parent_id": 0,
                    "child_index": 3,
                    "depth": 1,
                    "name": "t",
                    "byte_start": 32,
                    "byte_length": 4,
                    "value_byte_start": 35,
                    "value_byte_length": 1,
                    "type_byte_hex": "08",
                    "kind": "boolean",
                    "value": True,
                    "scalar_payload_hex": None,
                },
                {
                    "id": 5,
                    "parent_id": 0,
                    "child_index": 4,
                    "depth": 1,
                    "name": "z",
                    "byte_start": 36,
                    "byte_length": 3,
                    "value_byte_start": 39,
                    "value_byte_length": 0,
                    "type_byte_hex": "0a",
                    "kind": "null",
                    "value": None,
                    "scalar_payload_hex": None,
                },
                {
                    "id": 6,
                    "parent_id": 0,
                    "child_index": 5,
                    "depth": 1,
                    "name": "bin",
                    "byte_start": 39,
                    "byte_length": 12,
                    "value_byte_start": 44,
                    "value_byte_length": 7,
                    "type_byte_hex": "05",
                    "kind": "binary",
                    "value": "dead",
                    "scalar_payload_hex": None,
                    "binary_subtype": 0,
                },
                {
                    "id": 7,
                    "parent_id": 0,
                    "child_index": 6,
                    "depth": 1,
                    "name": "re",
                    "byte_start": 51,
                    "byte_length": 11,
                    "value_byte_start": 55,
                    "value_byte_length": 7,
                    "type_byte_hex": "0b",
                    "kind": "regex",
                    "value": "a.c",
                    "scalar_payload_hex": None,
                    "regex_options": "im",
                },
                {
                    "id": 8,
                    "parent_id": 0,
                    "child_index": 7,
                    "depth": 1,
                    "name": "sub",
                    "byte_start": 62,
                    "byte_length": 21,
                    "value_byte_start": 67,
                    "value_byte_length": 16,
                    "type_byte_hex": "03",
                    "kind": "document",
                    "value": None,
                    "scalar_payload_hex": None,
                    "child_count": 1,
                },
                {
                    "id": 9,
                    "parent_id": 8,
                    "child_index": 0,
                    "depth": 2,
                    "name": "j",
                    "byte_start": 71,
                    "byte_length": 11,
                    "value_byte_start": 74,
                    "value_byte_length": 8,
                    "type_byte_hex": "12",
                    "kind": "int64",
                    "value": "-5",
                    "scalar_payload_hex": "fbffffffffffffff",
                },
                {
                    "id": 10,
                    "parent_id": 0,
                    "child_index": 8,
                    "depth": 1,
                    "name": "arr",
                    "byte_start": 83,
                    "byte_length": 24,
                    "value_byte_start": 88,
                    "value_byte_length": 19,
                    "type_byte_hex": "04",
                    "kind": "array",
                    "value": None,
                    "scalar_payload_hex": None,
                    "child_count": 2,
                },
                {
                    "id": 11,
                    "parent_id": 10,
                    "child_index": 0,
                    "depth": 2,
                    "name": "0",
                    "byte_start": 92,
                    "byte_length": 7,
                    "value_byte_start": 95,
                    "value_byte_length": 4,
                    "type_byte_hex": "10",
                    "kind": "int32",
                    "value": "1",
                    "scalar_payload_hex": "01000000",
                },
                {
                    "id": 12,
                    "parent_id": 10,
                    "child_index": 1,
                    "depth": 2,
                    "name": "1",
                    "byte_start": 99,
                    "byte_length": 7,
                    "value_byte_start": 102,
                    "value_byte_length": 4,
                    "type_byte_hex": "10",
                    "kind": "int32",
                    "value": "2",
                    "scalar_payload_hex": "02000000",
                },
            ],
            "total_nodes": 13,
            "maximum_depth": 2,
            "document_count": 2,
            "array_count": 1,
            "regex_count": 1,
            "binary_count": 1,
            "offset": 0,
            "has_more": False,
            "full_source_validated": True,
            "regex_syntax_validated": False,
            "database_insert_compatibility_verified": False,
            "source_bytes": 108,
            "source_sha256": "9da2e3815e64823e0d8fefdf0f5c1396f2f89a5603540b8033702d39a49f421f",
        },
        "approx": False,
        "note": (
            "SYNTHETIC 108-byte BSON document assembled with struct as <int32 108> + nine "
            "elements + NUL. Element lengths are 1 (type) + len(name)+1 (cstring) + payload: "
            "i 7, s 10, d 11, t 4, z 3, bin 12, re 11, sub 21, arr 24 = 103, so with the "
            "4-byte length and 1-byte terminator the document is 108 bytes. Cumulative starts "
            "4,11,21,32,36,39,51,62,83 follow; value_byte_start skips type+name and "
            "value_byte_length is the remainder. The 16-byte 'sub' document puts 'j' at 67+4=71 "
            "and the 19-byte 'arr' puts '0'/'1' at 88+4=92 and 99. Preorder ids 0..12 give "
            "total_nodes 13 and maximum_depth 2; the root and 'sub' are the two documents, "
            "'arr' the one array, one regex and one binary. int32 7 = 07000000, int64 -5 = "
            "fbffffffffffffff, struct.pack('<d',1.5) = 000000000000f83f whose float.hex is "
            "0x1.8000000000000p+0."
        ),
    },
    "plugins.inspect_artifact_bundle": {
        "arguments": {"path": "manifest.json"},
        "files": {
            "manifest.json": (
                '{"version":1,"artifacts":[{"path":"payload.bin","size":3,"sha256":'
                '"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},'
                '{"path":"absent.bin","size":1,"sha256":'
                '"0000000000000000000000000000000000000000000000000000000000000000"}]}'
            ),
            "payload.bin": "abc",
        },
        "expect": {
            "manifest_version": 1,
            "artifact_count": 2,
            "declared_total_bytes": 4,
            "observations": [
                {
                    "index": 0,
                    "path": "payload.bin",
                    "declared_size": 3,
                    "declared_sha256": (
                        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                    ),
                    "status": "match",
                    "observed_size": 3,
                    "observed_sha256": (
                        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                    ),
                    "size_matches": True,
                    "sha256_matches": True,
                    "bytes_consumed": 3,
                    "error_class": None,
                },
                {
                    "index": 1,
                    "path": "absent.bin",
                    "declared_size": 1,
                    "declared_sha256": "0" * 64,
                    "status": "missing",
                    "observed_size": None,
                    "observed_sha256": None,
                    "size_matches": None,
                    "sha256_matches": None,
                    "bytes_consumed": 0,
                    "error_class": "FileNotFoundError",
                },
            ],
            "status_counts": {"match": 1, "missing": 1},
            "fully_read_count": 1,
            "matched_count": 1,
            "complete": False,
            "all_declarations_match": None,
            "manifest_has_artifacts": True,
            "aggregate_bytes_consumed": 3,
            "read_budget_bytes": 64_000_001,
            "expected_bindings_sha256": (
                "5a212b1e5b69c497d3457e84c194d865820dc8095087f6a6dbbe04f125b570b2"
            ),
            "observed_bindings_sha256": (
                "72c3c7e8ed1018d2a4a7d27959f183aa40bed4d84b424c986855295e87b8ab30"
            ),
            "unlisted_artifacts_checked": False,
            "research_certified": False,
            "source_bytes": 242,
            "source_sha256": "b5869d923182e0d74b594077af0636013e5440709ce27eeeec92c383a848f9fe",
        },
        "approx": False,
        "note": (
            "SYNTHETIC manifest, not a repo artifact. ba7816bf... is the standard SHA-256 of "
            "the three bytes 'abc', so payload.bin matches and consumes 3 bytes; absent.bin is "
            "never written, so its contained open raises FileNotFoundError and its observation "
            "carries no size, hash or match verdict. declared_total_bytes is 3+1 = 4 but only "
            "3 were consumed. One of two declarations matched, so complete is False and, because "
            "fully_read_count 1 != 2, all_declarations_match is None rather than False. Both "
            "binding digests are SHA-256 over compact sorted-key UTF-8 JSON of the two "
            "path-sorted triples: the expected one from declared (size,sha256) and the observed "
            "one from (3,hash) plus (null,null), which is why they differ."
        ),
    },
    "plugins.inspect_audit_ledger": {
        "arguments": {
            "path": "ledger.jsonl",
            "first_index": 0,
            "expected_previous_hash": "0" * 64,
            "expected_final_hash": "79c2f535aa4a4233631eb6d16aa10d7af89d47a674d863dda9019eaf693daa62",
            "expected_entry_count": 2,
            "offset": 0,
            "limit": 50,
        },
        # Each line is sorted compact ASCII JSON of the entry; entry_hash is
        # SHA-256 of the same encoding of the body without entry_hash.
        "files": {
            "ledger.jsonl": (
                '{"entry_hash":"c3bb28f2a6cacefdc61a030825737414f7d3568bd196ef5cda08e38053e6265c",'
                '"index":0,"kind":"research_run","payload":{"level":0.5,"pinball":0.25},'
                '"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000",'
                '"recorded_at":"2020-01-01T00:00:00Z","v":1}\n'
                '{"entry_hash":"79c2f535aa4a4233631eb6d16aa10d7af89d47a674d863dda9019eaf693daa62",'
                '"index":1,"kind":"paper_decision","payload":{"decision":"accept"},'
                '"prev_hash":"c3bb28f2a6cacefdc61a030825737414f7d3568bd196ef5cda08e38053e6265c",'
                '"recorded_at":"2020-01-01T00:01:00Z","v":1}\n'
            )
        },
        "expect": {
            "verification_scope": "supplied_v1_entry_chain_only",
            "checkpoint_signatures_verified": False,
            "research_eligibility_verified": False,
            "assessment": "consistent",
            "chain_consistent": True,
            "first_index": 0,
            "physical_line_count": 2,
            "structurally_parsed_entries": 2,
            "first_declared_hash": (
                "c3bb28f2a6cacefdc61a030825737414f7d3568bd196ef5cda08e38053e6265c"
            ),
            "last_declared_hash": "79c2f535aa4a4233631eb6d16aa10d7af89d47a674d863dda9019eaf693daa62",
            "expected_previous_hash": "0" * 64,
            "expected_final_hash_matches": True,
            "expected_entry_count_matches": True,
            "kind_counts": {"paper_decision": 1, "research_run": 1},
            "issue_counts": {},
            "issue_count": 0,
            "diagnostics": [],
            "omitted_diagnostics": 0,
            "entries": [
                {
                    "physical_line": 1,
                    "declared_index": 0,
                    "expected_index": 0,
                    "kind": "research_run",
                    "recorded_at": "2020-01-01T00:00:00Z",
                    "payload_field_count": 2,
                    "declared_entry_hash": (
                        "c3bb28f2a6cacefdc61a030825737414f7d3568bd196ef5cda08e38053e6265c"
                    ),
                    "calculated_entry_hash": (
                        "c3bb28f2a6cacefdc61a030825737414f7d3568bd196ef5cda08e38053e6265c"
                    ),
                    "entry_hash_matches": True,
                    "previous_link_matches": True,
                    "canonical_line": True,
                },
                {
                    "physical_line": 2,
                    "declared_index": 1,
                    "expected_index": 1,
                    "kind": "paper_decision",
                    "recorded_at": "2020-01-01T00:01:00Z",
                    "payload_field_count": 1,
                    "declared_entry_hash": (
                        "79c2f535aa4a4233631eb6d16aa10d7af89d47a674d863dda9019eaf693daa62"
                    ),
                    "calculated_entry_hash": (
                        "79c2f535aa4a4233631eb6d16aa10d7af89d47a674d863dda9019eaf693daa62"
                    ),
                    "entry_hash_matches": True,
                    "previous_link_matches": True,
                    "canonical_line": True,
                },
            ],
            "offset": 0,
            "next_offset": None,
            "source_sha256": "35f47b78a80602ffbd03a92d50157450ca5ce9de05df56f4587a662d29730f3e",
            "source_bytes": 545,
        },
        "approx": False,
        "note": (
            "SYNTHETIC two-entry v1 ledger invented for this case, not copied from any repo "
            "ledger. Using the module's documented convention -- body is the entry minus "
            "entry_hash, encoded as sorted compact ASCII-escaped JSON -- hashlib.sha256 gives "
            "c3bb28f2... for entry 0 (genesis prev_hash of 64 zeros) and 79c2f535... for entry 1 "
            "whose prev_hash is c3bb28f2..., so both digests and the single link agree. Each "
            "physical line is that same canonical encoding of its whole entry, 274 and 269 "
            "bytes plus one LF each, hence source_bytes 545 and canonical_line True on both. "
            "Indices 0 and 1 equal first_index + line - 1; payloads hold 2 and 1 fields; the "
            "supplied final-hash and entry-count anchors both match, so no issue is raised and "
            "next_offset is None because min(2, 0+50) is not below the 2-line file."
        ),
    },
    "plugins.inspect_research_receipt": {
        "arguments": {
            "path": "receipt.json",
            "seal_preimage": "receipt_body",
            "pointers": [
                "/verdict",
                "/payload/pinball",
                "/code_files/synthetic_module.py",
                "/missing/key",
            ],
        },
        "files": {
            "receipt.json": (
                '{"code_files":{"synthetic_module.py":'
                '"ebe4b027c34f4c6213cd4166b53e36059df74665192c0ee46e58cba23e4f4f0b"},'
                '"code_sha256":"d4f0ce05f084af14195fbcff752162216e84ebfafe3f11bf16ecbf7442e2cf3e",'
                '"data_label":"SYNTHETIC",'
                '"environment":{"fingerprint_sha256":'
                '"d7ec36036b3cf4e2004391e7f9ab8862c97e3ce9c0bf657e20a4b6702328bec7",'
                '"python_version":"3.12"},'
                '"kind":"synthetic_fixture","live_pnl_claim":false,"payload":{"pinball":0.25},'
                '"receipt_sha256":"ee40ee6b433240c6602cd89a6334fdbf734136be4d52375f0115ac79b30a6b58",'
                '"schema":"fx1.receipt/v1","verdict":"pass"}'
            )
        },
        "expect": {
            "inspection_scope": "declarations_and_digest_agreement_only",
            "research_eligibility_verified": False,
            "source_sha256": "d2b560851969c224c7ee313f3dd9afe2e7241f5dbdb5b27b432387d5ada6ec80",
            "source_bytes": 543,
            "declared_schema": "fx1.receipt/v1",
            "declared_kind": "synthetic_fixture",
            "declared_data_label": "SYNTHETIC",
            "declared_verdict": "pass",
            "declared_live_pnl_claim": False,
            "selected_seal_preimage": "receipt_body",
            "seal_excluded_top_level_fields": ["receipt_sha256"],
            "receipt_seal": {
                "status": "match",
                "declared_sha256": (
                    "ee40ee6b433240c6602cd89a6334fdbf734136be4d52375f0115ac79b30a6b58"
                ),
                "calculated_sha256": (
                    "ee40ee6b433240c6602cd89a6334fdbf734136be4d52375f0115ac79b30a6b58"
                ),
                "calculated_convention": "canonical_json",
                "matched_conventions": ["canonical_json", "strict_json"],
            },
            "environment_fingerprint": {
                "status": "match",
                "declared_sha256": (
                    "d7ec36036b3cf4e2004391e7f9ab8862c97e3ce9c0bf657e20a4b6702328bec7"
                ),
                "calculated_sha256": (
                    "d7ec36036b3cf4e2004391e7f9ab8862c97e3ce9c0bf657e20a4b6702328bec7"
                ),
                "calculated_convention": "canonical_json",
                "matched_conventions": ["canonical_json"],
            },
            "code_map_fingerprint": {
                "status": "match",
                "declared_sha256": (
                    "d4f0ce05f084af14195fbcff752162216e84ebfafe3f11bf16ecbf7442e2cf3e"
                ),
                "calculated_sha256": (
                    "d4f0ce05f084af14195fbcff752162216e84ebfafe3f11bf16ecbf7442e2cf3e"
                ),
                "calculated_convention": "canonical_json",
                "matched_conventions": ["canonical_json"],
            },
            "declared_code_file_count": 1,
            "payload_present": True,
            "selections": [
                {
                    "pointer": "/verdict",
                    "status": "found",
                    "value": "pass",
                    "canonical_value_sha256": (
                        "fedce8e8af485c2586f54c43a9b656dc70b41eb9be8c64b69c98391847dd4baf"
                    ),
                    "canonical_value_bytes": 6,
                },
                {
                    "pointer": "/payload/pinball",
                    "status": "found",
                    "value": 0.25,
                    "canonical_value_sha256": (
                        "a30a043314fa89294fa2c1c989a01fbb5329e5c085a5c5a8d27317656de24ae0"
                    ),
                    "canonical_value_bytes": 4,
                },
                {
                    "pointer": "/code_files/synthetic_module.py",
                    "status": "found",
                    "value": "ebe4b027c34f4c6213cd4166b53e36059df74665192c0ee46e58cba23e4f4f0b",
                    "canonical_value_sha256": (
                        "7dd7b9f53d36eb24925113be45fc7fc2ae3711f0eb4e4e866ad626b438aedc81"
                    ),
                    "canonical_value_bytes": 66,
                },
                {
                    "pointer": "/missing/key",
                    "status": "missing",
                    "value": None,
                    "canonical_value_sha256": None,
                    "canonical_value_bytes": None,
                },
            ],
        },
        "approx": False,
        "note": (
            "SYNTHETIC receipt written by hand; it is labeled data_label SYNTHETIC / "
            "kind synthetic_fixture and is not repo evidence. hashlib.sha256 over sorted "
            "compact UTF-8 JSON gives ebe4b027... for b'synthetic module bytes', "
            "d4f0ce05... for the one-entry code map, d7ec3603... for {'python_version':'3.12'} "
            "and ee40ee6b... for the whole body minus receipt_sha256. Every value is ASCII, so "
            "the UTF-8 and legacy ASCII-escaped encodings are byte-identical and the seal alone "
            "reports both conventions; environment and code map are checked only as "
            "canonical_json. Pointer canonical forms are '\"pass\"' (6 bytes), '0.25' (4 bytes) "
            "and the 64-hex digest in quotes (66 bytes); /missing/key resolves to nothing. The "
            "543-byte source is the compact single-line spelling above."
        ),
    },
    "skills.audit_resource_reservations": {
        "arguments": {
            "resources": [
                {
                    "resource_id": "pool",
                    "parent_resource_id": None,
                    "owner_id": "team",
                    "unit": "gpu_hour",
                    "capacity": 4,
                    "valid_from": "2020-01-01T00:00:00+00:00",
                    "valid_to": "2020-01-02T00:00:00+00:00",
                    "available_time": "2020-01-01T00:00:00+00:00",
                }
            ],
            "reservations": [
                {
                    "reservation_id": "r1",
                    "resource_id": "pool",
                    "owner_id": "team",
                    "quantity": 3,
                    "start": "2020-01-01T00:00:00+00:00",
                    "end": "2020-01-01T06:00:00+00:00",
                    "available_time": "2020-01-01T00:00:00+00:00",
                }
            ],
            "consumptions": [
                {
                    "consumption_id": "c1",
                    "reservation_id": "r1",
                    "owner_id": "team",
                    "quantity": 2,
                    "start": "2020-01-01T01:00:00+00:00",
                    "end": "2020-01-01T02:00:00+00:00",
                    "available_time": "2020-01-01T02:00:00+00:00",
                }
            ],
            "decision_time": "2020-01-01T12:00:00+00:00",
            "audit_start": "2020-01-01T00:00:00+00:00",
            "audit_end": "2020-01-02T00:00:00+00:00",
            "offset": 0,
            "limit": 100,
        },
        "expect": {
            "passed": True,
            "visible_declarations_complete": True,
            "resource_count": 1,
            "visible_resource_count": 1,
            "valid_resource_count": 1,
            "visible_reservation_count": 1,
            "included_reservation_count": 1,
            "excluded_reservation_count": 0,
            "visible_consumption_count": 1,
            "included_consumption_count": 1,
            "excluded_consumption_count": 0,
            "propagated_demand_count": 2,
            "issue_counts": {},
            "diagnostics": [],
            "omitted_diagnostics": 0,
            "resources": [
                {
                    "resource_row_index": 0,
                    "resource_id": "pool",
                    "ancestry_valid": True,
                    "ancestor_resource_row_indexes": [],
                    "declared_capacity": 4,
                    "reservations": {
                        "maximum_quantity": 3,
                        "quantity_microseconds": 64_800_000_000,
                        "excess_duration_microseconds": 0,
                        "excess_quantity_microseconds": 0,
                        "excess_span_count": 0,
                    },
                    "consumptions": {
                        "maximum_quantity": 2,
                        "quantity_microseconds": 7_200_000_000,
                        "excess_duration_microseconds": 0,
                        "excess_quantity_microseconds": 0,
                        "excess_span_count": 0,
                    },
                    "maximum_source_available_time": "2020-01-01T02:00:00Z",
                }
            ],
            "offset": 0,
            "has_more": False,
            "demand_scope": "explicit_audit_window_only",
            "declared_ownership_verified": False,
            "physical_consumption_verified": False,
        },
        "approx": False,
        "note": (
            "SYNTHETIC single-node capacity tree; no parent, so the ancestry path is just row 0 "
            "and ancestor_resource_row_indexes is empty. One reservation and one consumption "
            "propagate once each, giving propagated_demand_count 2. Sweeping the 86400-second "
            "window: the 3-unit reservation is active for 6 hours, so quantity_microseconds is "
            "3 * 6 * 3600 * 1e6 = 64_800_000_000 with peak 3; the 2-unit consumption is active "
            "for 1 hour, so 2 * 3600 * 1e6 = 7_200_000_000 with peak 2. Both peaks stay at or "
            "below capacity 4, so there is no excess and no issue. The latest source publication "
            "among the resource, reservation and consumption clocks is the consumption's 02:00."
        ),
    },
    "skills.audit_task_leases": {
        "arguments": {
            "events": [
                {
                    "task_id": "t",
                    "sequence": 0,
                    "event_time": "2020-01-01T00:00:00+00:00",
                    "available_time": "2020-01-01T00:00:00+00:00",
                    "action": "grant",
                    "worker_id": "w",
                    "epoch": 1,
                    "expires_time": "2020-01-01T02:00:00+00:00",
                },
                {
                    "task_id": "t",
                    "sequence": 1,
                    "event_time": "2020-01-01T01:00:00+00:00",
                    "available_time": "2020-01-01T01:00:00+00:00",
                    "action": "renew",
                    "worker_id": "w",
                    "epoch": 1,
                    "expires_time": "2020-01-01T03:00:00+00:00",
                },
            ],
            "claims": [
                {
                    "task_id": "t",
                    "worker_id": "w",
                    "epoch": 1,
                    "access_time": "2020-01-01T01:30:00+00:00",
                    "available_time": "2020-01-01T01:30:00+00:00",
                }
            ],
            "decision_time": "2020-01-01T01:30:00+00:00",
            "first_sequence": 0,
            "task_offset": 0,
            "task_limit": 100,
            "claim_offset": 0,
            "claim_limit": 100,
        },
        "expect": {
            "passed": True,
            "task_count": 1,
            "event_count": 2,
            "visible_event_count": 2,
            "future_event_count": 0,
            "claim_count": 1,
            "future_claim_count": 0,
            "declared_replay_rows": 4,
            "issue_counts": {},
            "diagnostics": [],
            "omitted_diagnostics": 0,
            "tasks": [
                {
                    "task_id": "t",
                    "history_consistent": True,
                    "projected_state": "active",
                    "active_worker_id": "w",
                    "active_epoch": 1,
                    "highest_legal_grant_epoch": 1,
                    "expires_time": "2020-01-01T03:00:00Z",
                    "last_legal_event_row_index": 1,
                    "grant_event_row_index": 0,
                    "first_fault_event_row_index": None,
                    "visible_event_count": 2,
                    "applied_event_count": 2,
                    "maximum_applied_available_time": "2020-01-01T01:00:00Z",
                }
            ],
            "task_offset": 0,
            "tasks_have_more": False,
            "claim_status_counts": {"consistent": 1},
            "claims": [
                {
                    "claim_row_index": 0,
                    "status": "consistent",
                    "active_worker_id": "w",
                    "active_epoch": 1,
                    "grant_event_row_index": 0,
                    "last_legal_event_row_index": 1,
                    "first_fault_event_row_index": None,
                    "maximum_lease_available_time": "2020-01-01T01:00:00Z",
                }
            ],
            "claim_offset": 0,
            "claims_have_more": False,
            "external_fencing_enforcement_verified": False,
            "worker_authentication_verified": False,
        },
        "approx": False,
        "note": (
            "SYNTHETIC two-event lease history. Sequence 0 is the first legal grant of epoch 1 "
            "with expiry 02:00 strictly after its 00:00 event clock; sequence 1 is a renewal by "
            "the same worker and epoch extending expiry to 03:00, which is strictly later, so "
            "both events apply and no fault is raised. At decision_time 01:30 the lease is "
            "active on [00:00, 03:00), hence projected_state active with worker w and epoch 1, "
            "last_legal_event_row_index 1 and grant_event_row_index 0. The single claim at "
            "01:30 is published at its own access time and declares exactly the active worker "
            "and epoch, so it is consistent. declared_replay_rows is a conservative charge, not "
            "a count of distinct replays: the 2 supplied event rows are charged once for the "
            "global replay, and the one distinct (task, access_time) claim view charges all 2 "
            "rows of that task again -- 2 + 2 = 4. The charge counts every row of the task "
            "whether or not it is decision-visible, and it is levied even though this claim's "
            "access_time equals decision_time so the view is served from cache."
        ),
    },
    "skills.reconstruct_order_book": {
        "arguments": {
            "stream_id": "syn",
            "events": [
                {
                    "sequence": 0,
                    "event_time": "2020-01-01T00:00:00+00:00",
                    "available_time": "2020-01-01T00:00:00+00:00",
                    "action": "add",
                    "order_id": "A",
                    "side": "bid",
                    "price_ticks": 100,
                    "quantity": 5,
                },
                {
                    "sequence": 1,
                    "event_time": "2020-01-01T00:00:01+00:00",
                    "available_time": "2020-01-01T00:00:01+00:00",
                    "action": "add",
                    "order_id": "B",
                    "side": "bid",
                    "price_ticks": 100,
                    "quantity": 7,
                },
                {
                    "sequence": 2,
                    "event_time": "2020-01-01T00:00:02+00:00",
                    "available_time": "2020-01-01T00:00:02+00:00",
                    "action": "add",
                    "order_id": "C",
                    "side": "ask",
                    "price_ticks": 101,
                    "quantity": 4,
                },
                {
                    "sequence": 3,
                    "event_time": "2020-01-01T00:00:03+00:00",
                    "available_time": "2020-01-01T00:00:03+00:00",
                    "action": "execute",
                    "order_id": "A",
                    "quantity": 2,
                },
                {
                    "sequence": 4,
                    "event_time": "2020-01-01T00:00:04+00:00",
                    "available_time": "2020-01-01T00:00:04+00:00",
                    "action": "replace",
                    "order_id": "B",
                    "price_ticks": 99,
                    "quantity": 3,
                },
            ],
            "snapshots": [{"decision_time": "2020-01-01T00:00:10+00:00"}],
            "first_sequence": 0,
            "execution_policy": "fifo_at_price",
            "order_offset": 0,
            "order_limit": 25,
            "level_offset": 0,
            "level_limit": 25,
        },
        "expect": {
            "passed": True,
            "stream_id": "syn",
            "execution_policy": "fifo_at_price",
            "declared_replay_rows": 5,
            "issue_counts": {},
            "diagnostics": [],
            "omitted_diagnostics": 0,
            "snapshots": [
                {
                    "snapshot_row_index": 0,
                    "status": "observed_prefix",
                    "visible_selected_row_count": 5,
                    "applied_row_count": 5,
                    "last_applied_sequence": 4,
                    "last_applied_event_time": "2020-01-01T00:00:04Z",
                    "maximum_applied_available_time": "2020-01-01T00:00:04Z",
                    "blocked_at_sequence": None,
                    "active_order_count": 3,
                    "price_level_count": 3,
                    "bid_quantity": 6,
                    "ask_quantity": 4,
                    "best_bid_ticks": 100,
                    "best_ask_ticks": 101,
                    "crossed_or_locked": False,
                    "executed_quantity": 2,
                    "reduced_quantity": 0,
                    "deleted_quantity": 0,
                    "orders": [
                        {
                            "order_id": "A",
                            "side": "bid",
                            "price_ticks": 100,
                            "remaining_quantity": 3,
                            "original_add_row_index": 0,
                            "priority_event_row_index": 0,
                            "last_event_row_index": 3,
                            "priority_sequence": 0,
                            "fifo_position_at_price": 0,
                        },
                        {
                            "order_id": "B",
                            "side": "bid",
                            "price_ticks": 99,
                            "remaining_quantity": 3,
                            "original_add_row_index": 1,
                            "priority_event_row_index": 4,
                            "last_event_row_index": 4,
                            "priority_sequence": 4,
                            "fifo_position_at_price": 0,
                        },
                        {
                            "order_id": "C",
                            "side": "ask",
                            "price_ticks": 101,
                            "remaining_quantity": 4,
                            "original_add_row_index": 2,
                            "priority_event_row_index": 2,
                            "last_event_row_index": 2,
                            "priority_sequence": 2,
                            "fifo_position_at_price": 0,
                        },
                    ],
                    "order_offset": 0,
                    "orders_have_more": False,
                    "levels": [
                        {
                            "side": "bid",
                            "price_ticks": 100,
                            "total_quantity": 3,
                            "order_count": 1,
                            "first_order_id": "A",
                            "first_priority_event_row_index": 0,
                        },
                        {
                            "side": "bid",
                            "price_ticks": 99,
                            "total_quantity": 3,
                            "order_count": 1,
                            "first_order_id": "B",
                            "first_priority_event_row_index": 4,
                        },
                        {
                            "side": "ask",
                            "price_ticks": 101,
                            "total_quantity": 4,
                            "order_count": 1,
                            "first_order_id": "C",
                            "first_priority_event_row_index": 2,
                        },
                    ],
                    "level_offset": 0,
                    "levels_have_more": False,
                }
            ],
            "initial_state": "declared_empty",
            "external_feed_completeness_verified": False,
            "venue_protocol_verified": False,
        },
        "approx": False,
        "note": (
            "SYNTHETIC five-event order log replayed from a declared empty book with one "
            "snapshot and no through_sequence, so the status is observed_prefix and "
            "declared_replay_rows is 5 * 1. Adds A and B share bid 100 (5 and 7) and C is the "
            "only ask at 101. Executing 2 of A is legal under fifo_at_price because A is the "
            "oldest order at 100, leaving A with 3 and making level 100 total 3; executed "
            "quantity is 2 and nothing is reduced or deleted. Replacing B with price 99 and "
            "quantity 3 keeps its side and original add row 1 but moves its priority to row 4 "
            "and sequence 4, emptying level 100 of B and creating level 99. The level sort key "
            "is (side == ask, -price for bids else price), so all bids come first in "
            "price-descending order and asks last in ascending order: levels 100, 99, 101. The "
            "order page walks those same levels in that same order and each level holds exactly "
            "one surviving order, so it is A, B, C. bid_quantity is 3 + 3 = 6, ask_quantity 4, "
            "best bid 100 is "
            "below best ask 101 so the book is neither crossed nor locked, and the last applied "
            "sequence is 4 at event clock 00:00:04."
        ),
    },
}
