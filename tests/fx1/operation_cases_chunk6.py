# Hand-checked FX-1 archive/container inspection cases. Expectation dicts match
# Output.model_dump(mode="json") and may be strict subsets of it. No fx1 import.
#
# Every fixture below is byte-stable: timestamps, ownership, modes and compression
# are pinned, and the bytes are embedded as base64 so a case never depends on how
# this file was produced. Every expected number was derived from an independent
# oracle BEFORE the operation ran -- hashlib.sha256 and len() over the fixture
# bytes, stdlib zipfile/tarfile/gzip/sqlite3 read-backs, struct.unpack over the
# safetensors prefix, and a hand walk of the Parquet footer's Compact-Thrift
# fields. None of them comes from an fx-1 code path or a captured run.

CASES = {
    "plugins.inspect_zip": {
        "arguments": {"path": "tiny.zip", "offset": 0, "limit": 50},
        "files": {
            "tiny.zip": {
                "base64": "UEsDBBQAAAAAAAAAIQCGphA2BQAAAAUAAAAFAAAAYS50eHRoZWxsb1BLAwQUAAAAAAAAACEAbUiDngIAAAACAAAABQAAAGIudHh0YWJQSwECFAMUAAAAAAAAACEAhqYQNgUAAAAFAAAABQAAAAAAAAAAAAAApAEAAAAAYS50eHRQSwECFAMUAAAAAAAAACEAbUiDngIAAAACAAAABQAAAAAAAAAAAAAAgAEoAAAAYi50eHRQSwUGAAAAAAIAAgBmAAAATQAAAAAA"
            }
        },
        "expect": {
            "entry_count": 2,
            "file_count": 2,
            "directory_count": 0,
            "symlink_count": 0,
            "encrypted_entry_count": 0,
            "duplicate_name_count": 0,
            "entries_with_path_issues": 0,
            "path_issue_counts": {},
            "total_compressed_bytes": 7,
            "total_declared_uncompressed_bytes": 7,
            "members": [
                {
                    "index": 0,
                    "name": "a.txt",
                    "original_name": "a.txt",
                    "name_was_overridden": False,
                    "directory": False,
                    "symlink": False,
                    "encrypted": False,
                    "compression_method": 0,
                    "compressed_bytes": 5,
                    "declared_uncompressed_bytes": 5,
                    "declared_crc32": "3610a686",
                    "expansion_ratio": 1.0,
                    "duplicate_name": False,
                    "path_issues": [],
                    "original_path_issues": [],
                },
                {
                    "index": 1,
                    "name": "b.txt",
                    "original_name": "b.txt",
                    "name_was_overridden": False,
                    "directory": False,
                    "symlink": False,
                    "encrypted": False,
                    "compression_method": 0,
                    "compressed_bytes": 2,
                    "declared_uncompressed_bytes": 2,
                    "declared_crc32": "9e83486d",
                    "expansion_ratio": 1.0,
                    "duplicate_name": False,
                    "path_issues": [],
                    "original_path_issues": [],
                },
            ],
            "offset": 0,
            "has_more": False,
            "source_sha256": "5f1a5df63a473fed258a8ea2ef60a0a49696c5996160e27198d0c9d80b2022a5",
            "source_bytes": 201,
        },
        "approx": False,
        "note": (
            "Two ZIP_STORED members pinned to 1980-01-01. zipfile read-back gives "
            "compress_size == file_size (5 + 2 = 7 bytes each way), so both "
            "expansion_ratios are exactly 1.0; zlib.crc32 independently yields "
            "0x3610a686 for b'hello' and 0x9e83486d for b'ab'. flag_bits 0 means "
            "unencrypted and external_attr 0o644 << 16 is not S_IFLNK, so no "
            "symlinks. sha256/len over the 201 fixture bytes give the digest "
            "and size."
        ),
    },
    "plugins.inspect_gzip": {
        "arguments": {"path": "tiny.gz", "max_uncompressed_bytes": 1000000},
        "files": {
            "tiny.gz": {
                "base64": "H4sIAAAAAAAAE8tIzcnJBwCGphA2BQAAAB+LCAAAAAAAABNLTAIAbUiDngIAAAA="
            }
        },
        "expect": {
            "member_count": 2,
            "compressed_bytes": 47,
            "uncompressed_bytes": 7,
            "expansion_ratio": 0.14893617021276595,
            "max_uncompressed_bytes": 1000000,
            "stream_validated": True,
            "crc_and_size_validated": True,
            "trailing_policy": "reject_all_nonmember_bytes",
            "concatenated_members_allowed": True,
            "members": [
                {
                    "index": 0,
                    "compressed_offset": 0,
                    "compressed_bytes": 25,
                    "uncompressed_bytes": 5,
                    "compressed_sha256": "8f00197768244e7fc925e2f92742e8aa0e3bf356323061813acc45748d4e98ec",
                    "uncompressed_sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                },
                {
                    "index": 1,
                    "compressed_offset": 25,
                    "compressed_bytes": 22,
                    "uncompressed_bytes": 2,
                    "compressed_sha256": "157cd00da5225e6ae69519d05f691e9eb27201e9d01fe9e547c181d6122a7dc6",
                    "uncompressed_sha256": "fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603",
                },
            ],
            "source_sha256": "fda709ff626e9b90cd1e0f70d425db08f7621f56f1adf88d0d012efcf86bc859",
            "uncompressed_sha256": "d490cbb6fff1654bcce0b23557c04ddfa87dc8bda6ff84f711f725f83dd90b39",
        },
        "approx": True,
        "note": (
            "The fixture is gzip.compress(b'hello') concatenated with "
            "gzip.compress(b'ab'), both mtime=0, so the member split is known "
            "without any decoder: offsets 0 and 25, lengths 25 and "
            "22 summing to 47. hashlib.sha256 over each slice gives "
            "the compressed digests; sha256 of b'hello', b'ab' and b'helloab' gives "
            "the three uncompressed digests. expansion_ratio is the quotient 7/47."
        ),
    },
    "plugins.inspect_tar": {
        "arguments": {"path": "tiny.tar", "offset": 0, "limit": 50},
        "files": {
            "tiny.tar": {
                "base64": "YS50eHQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADAwMDA2NDQAMDAwMDAwMAAwMDAwMDAwADAwMDAwMDAwMDA1ADAwMDAwMDAwMDAwADAwNjcyMQAgMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB1c3RhcgAwMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABoZWxsbwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHN1Yi8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwMDAwNzU1ADAwMDAwMDAAMDAwMDAwMAAwMDAwMDAwMDAwMAAwMDAwMDAwMDAwMAAwMDY1MzYAIDUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAdXN0YXIAMDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
            }
        },
        "expect": {
            "member_count": 2,
            "members": [
                {
                    "index": 0,
                    "header_offset": 0,
                    "format": "ustar",
                    "name": "a.txt",
                    "kind": "regular",
                    "size": 5,
                    "mode": 420,
                    "uid": 0,
                    "gid": 0,
                    "mtime_unix_seconds": 0,
                    "link_target": None,
                    "device_major": None,
                    "device_minor": None,
                    "path_issues": [],
                    "link_target_issues": [],
                    "duplicate_name": False,
                    "header_sha256": "45e6a8bf5d38bd47270a184a49a4a7b67e34174d6344e2c2308cc7cc7e4ed142",
                    "payload_sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                },
                {
                    "index": 1,
                    "header_offset": 1024,
                    "format": "ustar",
                    "name": "sub/",
                    "kind": "directory",
                    "size": 0,
                    "mode": 493,
                    "uid": 0,
                    "gid": 0,
                    "mtime_unix_seconds": 0,
                    "link_target": None,
                    "device_major": None,
                    "device_minor": None,
                    "path_issues": [],
                    "link_target_issues": [],
                    "duplicate_name": False,
                    "header_sha256": "867c408e033a12d3f202a0a35f47347c3a610d1d80149b99df02c1ef53860600",
                    "payload_sha256": None,
                },
            ],
            "offset": 0,
            "has_more": False,
            "type_counts": {"regular": 1, "directory": 1},
            "total_payload_bytes": 5,
            "duplicate_name_count": 0,
            "members_with_path_issues": 0,
            "end_padding_bytes": 1024,
            "archive_structure_validated": True,
            "extraction_performed": False,
            "source_bytes": 2560,
            "source_sha256": "4dc3ad8c8c8c14e032aafdc5429504b197bf9986e814a5f9e233a371207a94f3",
        },
        "approx": False,
        "note": (
            "USTAR archive of one 5-byte regular file and one empty directory, "
            "mtime/uid/gid zero, then exactly the two mandated 512-byte zero end "
            "blocks, so the layout is 512 header + 512 payload block + 512 header "
            "+ 1024 end padding = 2560 bytes and end_padding_bytes is 1024. "
            "sha256 of bytes [0:512] and [1024:1536] gives the header digests, "
            "sha256(b'hello') the only payload digest (a directory declares none), "
            "and octal 0000644/0000755 in the headers are decimal 420/493."
        ),
    },
    "plugins.inspect_sqlite": {
        "arguments": {
            "path": "sample.db",
            "table": "sample",
            "offset": 0,
            "limit": 10,
            "max_progress_callbacks": 500,
        },
        "files": {
            "sample.db": {
                "base64": "U1FMaXRlIGZvcm1hdCAzAAIAAQEAQCAgAAAAAgAAAAIAAAAAAAAAAAAAAAEAAAAEAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAC6VyQ0AAAABAaAAAaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABeAQcXGRkBgRd0YWJsZXNhbXBsZXNhbXBsZQJDUkVBVEUgVEFCTEUgc2FtcGxlKGlkIElOVEVHRVIsIGxhYmVsIFRFWFQsIHJhdGlvIFJFQUwsIHBheWxvYWQgQkxPQikNAAAAAgHlAAHtAeUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYCBQEAAAAIEQEFAQ8HEAd4P+AAAAAAAABBQg=="
            }
        },
        "expect": {
            "schema_objects": [
                {
                    "type": "table",
                    "name": "sample",
                    "table_name": "sample",
                    "root_page": 2,
                    "declared_sql": (
                        "CREATE TABLE sample(id INTEGER, label TEXT, ratio REAL, payload BLOB)"
                    ),
                    "ordinary_table_candidate": True,
                }
            ],
            "schema_object_count": 1,
            "ordinary_table_candidates": 1,
            "selected_table": "sample",
            "table_columns": [
                {
                    "index": 0,
                    "name": "id",
                    "declared_type": "INTEGER",
                    "not_null": False,
                    "default_expression": None,
                    "primary_key_position": 0,
                },
                {
                    "index": 1,
                    "name": "label",
                    "declared_type": "TEXT",
                    "not_null": False,
                    "default_expression": None,
                    "primary_key_position": 0,
                },
                {
                    "index": 2,
                    "name": "ratio",
                    "declared_type": "REAL",
                    "not_null": False,
                    "default_expression": None,
                    "primary_key_position": 0,
                },
                {
                    "index": 3,
                    "name": "payload",
                    "declared_type": "BLOB",
                    "not_null": False,
                    "default_expression": None,
                    "primary_key_position": 0,
                },
            ],
            "selected_columns": ["id", "label", "ratio", "payload"],
            "rows": [
                [
                    {"storage_class": "integer", "value": "7", "encoding": "decimal_string"},
                    {"storage_class": "text", "value": "x"},
                    {"storage_class": "real", "value": 0.5},
                    {"storage_class": "blob", "value": "4142", "encoding": "hex"},
                ],
                [
                    {"storage_class": "integer", "value": "8", "encoding": "decimal_string"},
                    {"storage_class": "null", "value": None},
                    {"storage_class": "null", "value": None},
                    {"storage_class": "null", "value": None},
                ],
            ],
            "returned_rows": 2,
            "offset": 0,
            "has_more": False,
            "total_table_rows": None,
            "row_order": "unordered_sqlite_table_scan",
            "page_size": 512,
            "page_count": 2,
            "text_encoding": "UTF-8",
            "source_bytes": 1024,
            "source_sha256": "732b100e85402608a3a6fe8084bb64587739eb0bcdae70c9d6675ac1c27611e8",
        },
        "approx": False,
        "note": (
            "A 512-byte-page database holding one table and two rows, so "
            "page_count is 1024/512 = 2 and the header page-count field reads "
            "2. A separate stdlib sqlite3 connection over the same bytes returns "
            "that single sqlite_schema row (root page 2), the four table_xinfo "
            "columns and the rows (7,'x',0.5,X'4142') and (8,NULL,NULL,NULL); the "
            "scan visits the single leaf page in rowid order. Integers become "
            "decimal strings and the two-byte blob becomes hex 4142."
        ),
    },
    "plugins.inspect_parquet": {
        "arguments": {
            "path": "tiny.parquet",
            "read_mode": "full",
            "max_footer_bytes": 2000000,
            "row_group_offset": 0,
            "row_group_limit": 20,
        },
        "files": {
            "tiny.parquet": {
                "base64": "UEFSMRUAFRwVHCwVBBUAFQYVBhwAAAACAAAABAEBAAAAAgAAABUAFSAVICwVBBUAFQYVBhwAAAACAAAABAEBAAAAeAEAAAB5FQIZPDUAGAZzY2hlbWEVBAAVAiUCGAR0aWNrABUMJQIYBWxhYmVsJQBMHAAAABYEGRwZLCYAHBUCGSUGABkYBHRpY2sVABYEFkIWQiYISRwVABUAFQIAPCkGGSYABAAAACYAHBUMGSUGABkYBWxhYmVsFQAWBBZGFkYmSkkcFQAVABUCADwWBBkGGSYABAAAABaIARYEJggWiAEAGRwYDEFSUk9XOnNjaGVtYRj4AS8vLy8vN0FBQUFBUUFBQUFBQUFLQUF3QUJnQUZBQWdBQ2dBQUFBQUJCQUFNQUFBQUNBQUlBQUFBQkFBSUFBQUFCQUFBQUFJQUFBQkVBQUFBQkFBQUFOVC8vLzhBQUFFRkVBQUFBQndBQUFBRUFBQUFBQUFBQUFVQUFBQnNZV0psYkFBQUFBUUFCQUFFQUFBQUVBQVVBQWdBQmdBSEFBd0FBQUFRQUJBQUFBQUFBQUVDRUFBQUFDQUFBQUFFQUFBQUFBQUFBQVFBQUFCMGFXTnJBQUFBQUFnQURBQUlBQWNBQ0FBQUFBQUFBQUVnQUFBQUFBQUFBQT09ABggcGFycXVldC1jcHAtYXJyb3cgdmVyc2lvbiAyNS4wLjEZLBwAABwAAADTAQAAUEFSMQ=="
            }
        },
        "expect": {
            "columns": [
                {"name": "tick", "logical_type": "int32", "nullable": True},
                {"name": "label", "logical_type": "string", "nullable": True},
            ],
            "leaf_column_count": 2,
            "total_rows": 2,
            "row_group_count": 1,
            "row_groups": [
                {
                    "index": 0,
                    "rows": 2,
                    "compressed_bytes": 68,
                    "uncompressed_bytes": 68,
                    "compression_codecs": ["UNCOMPRESSED"],
                }
            ],
            "row_group_offset": 0,
            "has_more_row_groups": False,
            "format_version": "1.0",
            "source_sha256": "f6a5c05dda415473f40bd87ba9e439b90bb457f9114e81d5cf927b9bfe8cf909",
            "source_bytes": 547,
            "read_mode": "full",
            "bytes_read": 547,
            "footer_offset": 72,
            "footer_bytes": 467,
            "footer_sha256": "4b7a6d9c8b8f43cea757b31b1a9fc96af906862d49a57daf09b0d36e75d70718",
            "footer_hash_scope": "serialized_file_metadata",
            "snapshot_guaranteed": False,
            "schema_origin": "parquet_types_without_serialized_arrow_origin",
            "serialized_arrow_schema_removed": True,
            "serialized_arrow_schema_restored": False,
        },
        "approx": False,
        "note": (
            "The whole 547-byte file was read back by hand, without pyarrow's "
            "reader: the last 4 bytes are PAR1 and the 4 before them are 467 "
            "little-endian, so footer_offset is 547-8-467 = 72, and hashlib.sha256 "
            "over the file and over footer[72:-8] gives both digests. A "
            "Compact-Thrift walk of that footer reads FileMetaData.version = 1 "
            "(hence '1.0'), num_rows = 2, one RowGroup with num_rows 2, and two "
            "ColumnChunks whose ColumnMetaData give codec enum 0 -- which "
            "parquet.thrift names UNCOMPRESSED, not pyarrow's writer spelling "
            "'NONE' -- plus total_compressed_size 33 and 35 and "
            "total_uncompressed_size 33 and 35, summing to 68 both ways and "
            "matching RowGroup.total_byte_size. Those column data pages start at 4 "
            "and 37, so 4+68 = 72 = footer_offset. The SchemaElements declare "
            "OPTIONAL INT32 'tick' and OPTIONAL BYTE_ARRAY 'label' with converted "
            "type UTF8, i.e. nullable int32 and string. The footer's only "
            "key/value key is ARROW:schema, so it is removed and never restored."
        ),
    },
    "plugins.inspect_safetensors": {
        "arguments": {
            "path": "tiny.safetensors",
            "read_mode": "full",
            "max_header_bytes": 1000000,
            "offset": 0,
            "limit": 50,
        },
        "files": {
            "tiny.safetensors": {
                "base64": "igAAAAAAAAB7Il9fbWV0YWRhdGFfXyI6eyJmb3JtYXQiOiJwdCJ9LCJhIjp7ImR0eXBlIjoiRjMyIiwic2hhcGUiOlsyXSwiZGF0YV9vZmZzZXRzIjpbMCw4XX0sImIiOnsiZHR5cGUiOiJJOCIsInNoYXBlIjpbM10sImRhdGFfb2Zmc2V0cyI6WzgsMTFdfX0AAIA/AAAAQAECAw=="
            }
        },
        "expect": {
            "tensor_count": 2,
            "scalar_tensor_count": 0,
            "empty_tensor_count": 0,
            "tensors": [
                {
                    "name": "a",
                    "dtype": "F32",
                    "shape": [2],
                    "elements": 2,
                    "bits_per_element": 32,
                    "payload_start": 0,
                    "payload_end": 8,
                    "absolute_file_start": 146,
                    "data_bytes": 8,
                },
                {
                    "name": "b",
                    "dtype": "I8",
                    "shape": [3],
                    "elements": 3,
                    "bits_per_element": 8,
                    "payload_start": 8,
                    "payload_end": 11,
                    "absolute_file_start": 154,
                    "data_bytes": 3,
                },
            ],
            "dtype_counts": {"F32": 1, "I8": 1},
            "metadata": {"format": "pt"},
            "offset": 0,
            "has_more": False,
            "payload_bytes": 11,
            "declared_layout_complete": True,
            "tensor_values_validated": False,
            "tensors_deserialized": False,
            "snapshot_guaranteed": False,
            "read_mode": "full",
            "bytes_read": 157,
            "header_bytes": 138,
            "header_sha256": "d11504700fcd7d45f01d1c15d25c512fefdd2b80afe7cb27ad6ce42176df0440",
            "header_hash_scope": "serialized_json_header_including_space_padding",
            "source_bytes": 157,
            "source_sha256": "308976c065b63868535ddb7eb8f43462c8470375baab4777d8daa500f4abf42d",
        },
        "approx": False,
        "note": (
            "Built by hand with struct: an 8-byte little-endian prefix of 138, "
            "that many bytes of JSON, then 11 payload bytes (two F32s and three I8s). "
            "struct.unpack('<Q') over the prefix re-reads 138, and "
            "157-8-138 = 11 is the payload. F32 is 32 bits so "
            "shape [2] is 2*32/8 = 8 bytes and I8 is 8 bits so shape [3] is 3 bytes, "
            "placing the tensors at [0,8) and [8,11) with absolute starts "
            "8+138 = 146 and 146+8 = 154. "
            "hashlib.sha256 over the header slice and the whole file gives both digests."
        ),
    },
}
