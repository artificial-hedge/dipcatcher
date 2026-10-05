# Hand-checked FX-1 invocation cases for the scientific/binary format inspectors.
# Expectation values match Output.model_dump(mode="json").
#
# Every fixture below is byte-exact synthetic data built by hand (struct.pack for the
# binary formats, literal text for NRRD and Zarr JSON). Every asserted number was
# derived from an oracle that is not the operation under test: hashlib.sha256 over the
# fixture bytes, len() of the fixture bytes, struct.unpack of bytes this file itself
# specifies, numpy's documented dtype string rules for the NPY descriptor, a
# transcription of Bob Jenkins' public-domain lookup3 (lookup3.c, May 2006) validated
# against that file's driver5() known-answer vectors for the HDF5 superblock checksum,
# pyarrow's own IPC reader plus buffer arithmetic for the Arrow stream, and plain
# Fraction/Decimal arithmetic for the STL geometry. Nothing here was copied from a
# previous run of these operations.

CASES = {
    "plugins.inspect_numpy_array": {
        "arguments": {"path": "tiny.npy"},
        # Hand-built NPY v1.0: b"\x93NUMPY" + b"\x01\x00" + u16 header length 118 +
        # the dict literal padded with spaces to make 10 + 118 = 128 (a multiple of 64)
        # and terminated by "\n" + 12 payload bytes. numpy.load() reads this file back
        # as shape (2, 3) dtype >i2 with values [[1, -2, 3], [-4, 5, -6]].
        "files": {
            "tiny.npy": {
                "base64": "k05VTVBZAQB2AHsnZGVzY3InOiAnPmkyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAf/+AAP//AAF//o="
            }
        },
        "expect": {
            "format_version": "1.0",
            "shape": [2, 3],
            "dimension_count": 2,
            "element_count": 6,
            "dtype": ">i2",
            "dtype_kind": "i",
            "original_descriptor": "'>i2'",
            "item_bytes": 2,
            "byte_order": ">",
            "fortran_order": False,
            "fields": [],
            "payload_offset": 128,
            "payload_bytes": 12,
            "source_sha256": "480d1e7cd27931414279226b5b8c806d5ec7f272e8df5580e945a8f02b53e420",
            "source_bytes": 140,
        },
        "approx": False,
        "note": (
            "Version bytes are 1,0 so format_version is '1.0'. The v1.0 header length is a "
            "2-byte little-endian field holding 118, so payload_offset is 8 + 2 + 118 = 128 "
            "(10 + 118 is a multiple of 64, matching numpy's own alignment). shape (2, 3) "
            "gives element_count 2 * 3 = 6 and dimension_count 2. The descriptor '>i2' is a "
            "big-endian 2-byte signed integer, so item_bytes is 2, dtype_kind is 'i' and "
            "byte_order is '>' (non-native order is never normalized away by numpy); "
            "original_descriptor is repr('>i2'). 6 elements * 2 bytes = 12 payload bytes and "
            "the file is 128 + 12 = 140 bytes, so payload_bytes is exactly 12. The dtype is "
            "not structured, so fields is empty. source_sha256 is SHA-256 over all 140 bytes."
        ),
    },
    "plugins.inspect_hdf5_superblock": {
        "arguments": {"path": "superblock.h5", "read_mode": "header"},
        # Hand-built HDF5 version-2 superblock at file offset 0 with 4-byte offset and
        # length fields: 8-byte signature, version/widths/flags, then base=0,
        # extension=0xFFFFFFFF (undefined), eof=48, root=40 and a 4-byte lookup3
        # checksum. 32 zero bytes follow the superblock so the 64-byte file extends past
        # the declared EOF. No HDF5 library is installed, and none is needed: the
        # inspector only reads these declared header fields.
        "files": {
            "superblock.h5": {
                "base64": "iUhERg0KGgoCBAQAAAAAAP////8wAAAAKAAAALegJxgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="
            }
        },
        "expect": {
            "superblock_version": 2,
            "signature_byte_start": 0,
            "superblock_bytes": 32,
            "signature_probes": 1,
            "offset_width": 4,
            "length_width": 4,
            "observed_consistency_flags": 0,
            "observed_write_access_flag": None,
            "observed_swmr_write_flag": None,
            "recorded_base_address": 0,
            "declared_absolute_eof": 48,
            "physical_bytes_after_declared_eof": 16,
            "root_group_relative_address": 40,
            "root_group_absolute_address": 40,
            "extension_relative_address": None,
            "extension_absolute_address": None,
            "stored_lookup3_checksum_hex": "1827a0b7",
            "checksum_matches": True,
            "object_graph_validated": False,
            "full_hdf5_validity_verified": False,
            "read_mode": "header",
            "bytes_read": 52,
            "superblock_sha256": "ecfee98254dceaee27b8e40bdb6f6112c663217e947d57a118d3c06b379d0f68",
            "superblock_hash_scope": "original_superblock_including_checksum",
            "source_bytes": 64,
            "source_sha256": None,
        },
        "approx": False,
        "note": (
            "The signature sits at file offset 0, so it is found on the first probe "
            "(signature_probes 1, signature_byte_start 0). Byte 8 is the superblock version 2, "
            "byte 9 the offset width 4 and byte 10 the length width 4, giving a superblock of "
            "16 + 4 * 4 = 32 bytes. Header mode reads three bounded ranges: an 8-byte probe, "
            "the 12-byte fixed prefix and the 32-byte superblock, so bytes_read is "
            "8 + 12 + 32 = 52. The four 4-byte little-endian address fields are base 0, "
            "extension 0xFFFFFFFF, eof 48 and root 40; 0xFFFFFFFF is the undefined sentinel "
            "for width 4, so both extension fields are null, and root_group_absolute_address "
            "is base + root = 0 + 40 = 40. physical_bytes_after_declared_eof is the 64-byte "
            "file minus eof 48 = 16. Version 2 does not interpret the consistency flag bits, "
            "so both v3-only flag observations are null. stored_lookup3_checksum_hex is the "
            "trailing 4 bytes read little-endian: 0x1827a0b7, computed independently with a "
            "transcription of Jenkins' public-domain hashlittle() over the first 28 bytes "
            "(that transcription reproduces lookup3.c driver5()'s published vectors "
            "deadbeef / 17770551 / cd628161). superblock_sha256 covers those 32 original "
            "bytes including the stored checksum; header mode never hashes the whole file, so "
            "source_sha256 is null."
        ),
    },
    "plugins.inspect_netcdf_classic": {
        "arguments": {"path": "tiny.nc", "read_mode": "header"},
        # Hand-built classic CDF1 file, big-endian throughout: magic, numrecs 0, one
        # 3-long dimension "row", one NC_INT global attribute "scale" = (-1, 2), and one
        # fixed NC_INT variable "x" over dimension 0. begin (108) is the first byte after
        # the header, and the file ends exactly at begin + vsize so there are no trailing
        # bytes. No netCDF4/h5py is installed; the format is simple enough to write with
        # struct.pack(">I", ...).
        "files": {
            "tiny.nc": {
                "base64": "Q0RGAQAAAAAAAAAKAAAAAQAAAANyb3cAAAAAAwAAAAwAAAABAAAABXNjYWxlAAAAAAAABAAAAAL/////AAAAAgAAAAsAAAABAAAAAXgAAAAAAAABAAAAAAAAAAAAAAAAAAAABAAAAAwAAABsAAAACv///+wAAAAe"
            }
        },
        "expect": {
            "format": "CDF1",
            "declared_record_count": 0,
            "dimensions": [
                {
                    "index": 0,
                    "name": "row",
                    "declared_length": 3,
                    "unlimited": False,
                    "resolved_length": 3,
                }
            ],
            "global_attributes": [
                {
                    "name": "scale",
                    "type": "NC_INT",
                    "element_count": 2,
                    "raw_hex": "ffffffff00000002",
                    "numeric_values": ["-1", "2"],
                    "encoding": "signed_decimal_strings",
                }
            ],
            "variable_count": 1,
            "fixed_variable_count": 1,
            "record_variable_count": 0,
            "record_start": None,
            "record_stride": None,
            "unpadded_single_record_variable": False,
            "uninterpreted_gap_bytes": 0,
            "variables": [
                {
                    "index": 0,
                    "name": "x",
                    "type": "NC_INT",
                    "dimension_ids": [0],
                    "shape": [3],
                    "attributes": [],
                    "record_variable": False,
                    "begin": 108,
                    "declared_vsize": 12,
                    "elements_per_slab": 3,
                    "data_bytes_per_slab": 12,
                    "total_elements": 3,
                    "record_stride": None,
                    "last_data_end_exclusive": 120,
                    "allocated_end_exclusive": 120,
                }
            ],
            "variable_offset": 0,
            "has_more_variables": False,
            "attribute_count": 1,
            "raw_attribute_bytes": 8,
            "numeric_attribute_values": 2,
            "declared_layout_checked": True,
            "variable_values_decoded": False,
            "variable_arrays_allocated": False,
            "payload_contents_validated": False,
            "read_mode": "header",
            "bytes_read": 120,
            "header_bytes": 108,
            "header_sha256": "bdcc8d05fa3b5278f2d3aec9bab1369b368824ff0317f8cc1f54d35436bc0367",
            "source_bytes": 120,
            "source_sha256": None,
        },
        "approx": False,
        "note": (
            "Magic b'CDF\\x01' selects CDF1, whose variable begin fields are 4 bytes wide. "
            "numrecs is 0 and there is no zero-length dimension, so declared_record_count is "
            "0 and no dimension is unlimited; dimension 0 'row' therefore resolves to its "
            "declared length 3. The single global attribute is NC_INT (type code 4) with 2 "
            "elements, so raw_attribute_bytes is 2 * 4 = 8 (already 4-byte aligned, no "
            "padding) and numeric_attribute_values is 2; the 8 raw bytes ffffffff00000002 "
            "read big-endian signed are -1 and 2. Counting the header by hand: 4 magic + 4 "
            "numrecs + 8 dim tag/count + 4 name length + 3 'row' + 1 pad + 4 length + 8 "
            "attribute tag/count + 4 name length + 5 'scale' + 3 pad + 8 type/nelems + 8 "
            "value + 8 variable tag/count + 4 name length + 1 'x' + 3 pad + 4 ndims + 4 "
            "dimid + 8 absent attribute list + 4 type + 4 vsize + 4 begin = 108 bytes, which "
            "is both header_bytes and the variable's begin, so uninterpreted_gap_bytes is 0. "
            "The variable is 3 int32 elements = 12 bytes, already a multiple of 4, so "
            "declared_vsize and data_bytes_per_slab are both 12 and "
            "last_data_end_exclusive = allocated_end_exclusive = 108 + 12 = 120. With no "
            "record variables the file must end exactly there, so source_bytes is 120. Header "
            "mode reads one prefix range of min(120, 2000000) = 120 bytes, hence bytes_read "
            "120 and source_sha256 null; header_sha256 is SHA-256 over the first 108 parsed "
            "bytes only."
        ),
    },
    "plugins.inspect_tiff_directory": {
        "arguments": {"path": "tiny.tif"},
        # Hand-built classic little-endian TIFF: b"II" + u16 42 + u32 first-IFD offset 8,
        # then one IFD holding three inline-valued tags (256 ImageWidth SHORT 2,
        # 273 StripOffsets LONG 52, 279 StripByteCounts LONG 4) and a null next-IFD
        # pointer. Two pad bytes separate the IFD end (50) from the 4-byte strip at 52,
        # making the file exactly 56 bytes. No IFD links, so the graph has no edges.
        "files": {
            "tiny.tif": {
                "base64": "SUkqAAgAAAADAAABAwABAAAAAgAAABEBBAABAAAANAAAABcBBAABAAAABAAAAAAAAAAAAAECAwQ="
            }
        },
        "expect": {
            "byte_order": "little",
            "first_ifd_byte_start": 8,
            "directories": [
                {
                    "source_byte_start": 8,
                    "byte_length": 42,
                    "entry_count": 3,
                    "next_ifd_byte_start": None,
                    "payload_reference_count": 1,
                    "summed_declared_payload_bytes": 4,
                    "payload_span_preview": [
                        {"kind": "strip", "index": 0, "source_byte_start": 52, "byte_length": 4}
                    ],
                    "payload_span_preview_truncated": False,
                }
            ],
            "edges": [],
            "shared_directory_reference_count": 0,
            "tag_count": 3,
            "typed_value_count": 3,
            "summed_value_bytes": 10,
            "payload_reference_count": 1,
            "tags": [
                {
                    "tag_index": 0,
                    "directory_byte_start": 8,
                    "entry_byte_start": 10,
                    "tag_id": 256,
                    "private_tag_number": False,
                    "type_name": "SHORT",
                    "value_count": 1,
                    "value_byte_start": 18,
                    "value_bytes": 2,
                    "inline": True,
                    "value_sha256": "99be5efb88ca2013bd8e4eb035fd42d5245468fe9afa70d8ba9c1c419a48c4e8",
                    "raw_preview_hex": "0200",
                    "raw_preview_truncated": False,
                    "preview_values": ["2"],
                    "preview_values_truncated": False,
                    "nonfinite_float_count": 0,
                    "zero_denominator_count": 0,
                    "meaning": "primitive_observation",
                },
                {
                    "tag_index": 1,
                    "directory_byte_start": 8,
                    "entry_byte_start": 22,
                    "tag_id": 273,
                    "private_tag_number": False,
                    "type_name": "LONG",
                    "value_count": 1,
                    "value_byte_start": 30,
                    "value_bytes": 4,
                    "inline": True,
                    "value_sha256": "77f906b94309dc84a1d71649eeac2d708182919f62d74580340943d6d8014bf9",
                    "raw_preview_hex": "34000000",
                    "raw_preview_truncated": False,
                    "preview_values": ["52"],
                    "preview_values_truncated": False,
                    "nonfinite_float_count": 0,
                    "zero_denominator_count": 0,
                    "meaning": "payload_offset_or_count",
                },
                {
                    "tag_index": 2,
                    "directory_byte_start": 8,
                    "entry_byte_start": 34,
                    "tag_id": 279,
                    "private_tag_number": False,
                    "type_name": "LONG",
                    "value_count": 1,
                    "value_byte_start": 42,
                    "value_bytes": 4,
                    "inline": True,
                    "value_sha256": "fb5e512425fc9449316ec95969ebe71e2d576dbab833d61e2a5b9330fd70ee02",
                    "raw_preview_hex": "04000000",
                    "raw_preview_truncated": False,
                    "preview_values": ["4"],
                    "preview_values_truncated": False,
                    "nonfinite_float_count": 0,
                    "zero_denominator_count": 0,
                    "meaning": "payload_offset_or_count",
                },
            ],
            "offset": 0,
            "has_more": False,
            "supported_metadata_graph_validated": True,
            "graph_has_cycle": False,
            "image_payload_decoded": False,
            "source_bytes": 56,
            "source_sha256": "2bae176271d9e33a5244b8eb37bc91b622d1ed22ea4a1a92a68ff99f7a2ceac9",
        },
        "approx": False,
        "note": (
            "Bytes 0-1 are b'II', so byte_order is little and all integers are read "
            "little-endian; bytes 2-3 are u16 42 and bytes 4-7 are u32 8, so "
            "first_ifd_byte_start is 8. The IFD's own u16 entry count at byte 8 is 3, giving "
            "byte_length 2 + 12 * 3 + 4 = 42 and spanning bytes 8..50. Each 12-byte entry "
            "starts at 10, 22 and 34, and because every value is at most 4 bytes it is "
            "stored inline at entry start + 8, i.e. value_byte_start 18, 30 and 42. Tag 256 "
            "is one SHORT (2 bytes, raw 0200 = 2); tags 273 and 279 are one LONG each "
            "(4 bytes, raw 34000000 = 52 and 04000000 = 4). typed_value_count is 1 + 1 + 1 "
            "= 3 and summed_value_bytes is 2 + 4 + 4 = 10. The next-IFD u32 at byte 46 is 0, "
            "so next_ifd_byte_start is null and there are no edges; "
            "shared_directory_reference_count is 0 edges - 1 directory + 1 = 0. Tags 273/279 "
            "are a paired strip offset/count, so payload_reference_count is 1 and the "
            "declared strip is 4 bytes at offset 52, which fits inside the 56-byte file "
            "(52 + 4 = 56) and starts at or after byte 8. All three tag numbers are below "
            "32768, so none is private; 256 is a plain primitive observation while 273/279 "
            "are payload references. With the default page (offset 0, limit 50) all 3 tags "
            "are returned, so has_more is 0 + 3 < 3 = False. Each value_sha256 is SHA-256 "
            "over that tag's raw value bytes alone, and source_sha256 is over all 56 bytes."
        ),
    },
    "plugins.read_arrow_ipc": {
        "arguments": {"path": "tiny.arrows"},
        # Arrow IPC stream (not file) written by pyarrow 25.0.1 for the schema
        # n: int32, x: float64 with a single 2-row batch holding n = [1, -2] and
        # x = [1.5, -0.25], followed by the explicit end-of-stream marker. Both float
        # values are exactly representable in binary, so no tolerance is needed.
        "files": {
            "tiny.arrows": {
                "base64": "/////6gAAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAIAAABEAAAABAAAANT///8AAAEDEAAAABgAAAAEAAAAAAAAAAEAAAB4AAYACAAGAAYAAAAAAAIAEAAUAAgABgAHAAwAAAAQABAAAAAAAAECEAAAABwAAAAEAAAAAAAAAAEAAABuAAAACAAMAAgABwAIAAAAAAAAASAAAAD/////uAAAABQAAAAAAAAADAAWAAYABQAIAAwADAAAAAADBAAYAAAAGAAAAAAAAAAAAAoAGAAMAAQACAAKAAAAXAAAABAAAAACAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAACAAAAAAAAAAQAAAAAAAAAAAAAAACAAAAAgAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAABAAAA/v///wAAAAAAAPg/AAAAAAAA0L//////AAAAAA=="
            }
        },
        "expect": {
            "format": "stream",
            "columns": [
                {
                    "name": "n",
                    "arrow_type": "int32",
                    "nullable": True,
                    "json_encoding": "integer_string",
                    "unit": None,
                    "timezone": None,
                },
                {
                    "name": "x",
                    "arrow_type": "double",
                    "nullable": True,
                    "json_encoding": "finite_number",
                    "unit": None,
                    "timezone": None,
                },
            ],
            "selected_columns": ["n", "x"],
            "rows": [{"n": "1", "x": 1.5}, {"n": "-2", "x": -0.25}],
            "total_rows": 2,
            "batch_count": 1,
            "offset": 0,
            "returned_rows": 2,
            "has_more": False,
            "observations_present": True,
            "all_columns_and_rows_validated": True,
            "compression_policy": "reject_before_decoding",
            "declared_buffer_bytes": 24,
            "source_bytes": 400,
            "source_sha256": "67cb9728fb7113ab20b994e15bb8dda1d61ba945e088a06fce651d7da9b11667",
        },
        "approx": False,
        "note": (
            "The bytes do not start with b'ARROW1', so this is the stream format; they end "
            "with the 8-byte continuation-plus-zero-length end-of-stream marker "
            "(ffffffff00000000) and nothing follows it. One schema message and one record "
            "batch message give batch_count 1, and the batch declares 2 rows, so total_rows "
            "and returned_rows are 2 and has_more is 0 + 2 < 2 = False. Both fields are "
            "nullable with no nulls. Column n is Arrow primitive type id 2 (integer), which "
            "encodes as an exact decimal string, so the rows carry '1' and '-2'; column x is "
            "type id 3 with precision 2 (float64, str(pa.float64()) == 'double'), which "
            "encodes as a finite JSON number, so the rows carry 1.5 and -0.25 exactly. "
            "Neither type is temporal, so unit and timezone are null. declared_buffer_bytes "
            "is the sum of the four declared buffer lengths: each field contributes a "
            "validity buffer of length 0 (no nulls) plus a values buffer of 2 * 4 = 8 bytes "
            "for int32 and 2 * 8 = 16 bytes for float64, i.e. 0 + 8 + 0 + 16 = 24. That "
            "matches the body directly: the packed little-endian bytes for [1, -2] and for "
            "[1.5, -0.25] each occur exactly once in the file, at contiguous offsets 368 and "
            "376, so the 24-byte body is [368, 392) and the last 8 bytes are the EOS marker. "
            "source_bytes is len() of the 400-byte fixture and source_sha256 its SHA-256."
        ),
    },
    # --- Below this line the three operations are not yet in the shipped registry ---
    "plugins.inspect_zarr_metadata": {
        "arguments": {"path": ".zarray"},
        # A Zarr v2 array metadata document with exactly the eight required keys plus the
        # optional dimension_separator. Written as literal JSON text (no zarr library is
        # installed, and the inspector deliberately never opens chunk data or codecs).
        "files": {
            ".zarray": (
                '{"zarr_format":2,"shape":[5,3],"chunks":[2,3],"dtype":"<i2",'
                '"compressor":{"id":"zlib","level":1},"fill_value":0,"order":"C",'
                '"filters":[{"id":"delta","dtype":"<i2"}],"dimension_separator":"/"}'
            )
        },
        "expect": {
            "zarr_format": 2,
            "shape": [5, 3],
            "chunk_shape": [2, 3],
            "dtype": "<i2",
            "dtype_kind": "signed_integer",
            "item_bytes": 2,
            "byte_order": "little",
            "chunk_memory_order": "C",
            "dimension_separator": "/",
            "logical_elements": "15",
            "logical_array_bytes": "30",
            "nominal_chunk_elements": "6",
            "nominal_chunk_bytes": "12",
            "chunk_grid_shape": [3, 1],
            "chunk_grid_count": "3",
            "array_is_empty": False,
            "fill": {
                "kind": "integer",
                "exact_declaration": "0",
                "dtype_quantization_performed": False,
            },
            "compressor": {
                "id": "zlib",
                "configuration_json": '{"id":"zlib","level":1}',
                "codec_parameters_validated": False,
            },
            "filters": [
                {
                    "id": "delta",
                    "configuration_json": '{"dtype":"<i2","id":"delta"}',
                    "codec_parameters_validated": False,
                }
            ],
            "chunk_page": [
                {
                    "grid_index": "0",
                    "coordinate": [0, 0],
                    "key": "0/0",
                    "logical_start": [0, 0],
                    "logical_stop": [2, 3],
                    "logical_shape": [2, 3],
                    "logical_elements": "6",
                    "nominal_elements": "6",
                    "overhang_elements": "0",
                },
                {
                    "grid_index": "1",
                    "coordinate": [1, 0],
                    "key": "1/0",
                    "logical_start": [2, 0],
                    "logical_stop": [4, 3],
                    "logical_shape": [2, 3],
                    "logical_elements": "6",
                    "nominal_elements": "6",
                    "overhang_elements": "0",
                },
                {
                    "grid_index": "2",
                    "coordinate": [2, 0],
                    "key": "2/0",
                    "logical_start": [4, 0],
                    "logical_stop": [5, 3],
                    "logical_shape": [1, 3],
                    "logical_elements": "3",
                    "nominal_elements": "6",
                    "overhang_elements": "3",
                },
            ],
            "offset": "0",
            "has_more": False,
            "chunk_page_order": "lexicographic_last_axis_fastest",
            "chunk_files_read": False,
            "codec_implementations_loaded": False,
            "metadata_subset_validated": True,
            "source_bytes": 191,
            "source_sha256": "85c06dbd61a535a80f6ec30c94df7710dc197c4da31df22c3ec4672fcfe7192f",
        },
        "approx": False,
        "note": (
            "zarr_format is the JSON integer 2, the only supported version. dtype '<i2' "
            "splits into endian '<', kind 'i' and width 2, so dtype_kind is signed_integer, "
            "item_bytes is 2 and byte_order is little (width 1 would be not_applicable). "
            "shape [5, 3] gives logical_elements 5 * 3 = 15 and logical_array_bytes "
            "15 * 2 = 30; chunks [2, 3] give nominal_chunk_elements 6 and "
            "nominal_chunk_bytes 12. The grid is ceil(5/2) by ceil(3/3) = [3, 1], so "
            "chunk_grid_count is 3 and array_is_empty is False. Enumerating lexicographically "
            "with the last axis fastest: index 0 -> coordinate [0, 0], index 1 -> [1, 0], "
            "index 2 -> [2, 0]; the separator is '/' so the keys are '0/0', '1/0', '2/0'. "
            "Chunk starts are coordinate * chunk length, i.e. [0,0], [2,0], [4,0], and stops "
            "are min(start + chunk, shape), i.e. [2,3], [4,3], [5,3] - the third chunk is "
            "clipped by the 5-long axis, so its logical_shape is [1, 3], logical_elements "
            "1 * 3 = 3 and overhang_elements 6 - 3 = 3, while the first two are full with "
            "overhang 0. fill_value is the JSON integer 0, which lies inside the int16 range "
            "-32768..32767, so the fill kind is integer with the exact declaration '0' and no "
            "quantization. Both codec objects keep their declared keys in sorted order as "
            "canonical JSON: {id, level} for the compressor and {dtype, id} for the delta "
            "filter, and neither codec is imported or parameter-validated. The default page "
            "(offset 0, limit 50) covers all 3 chunks, so has_more is 0 + 3 < 3 = False. "
            "source_bytes is the 191 UTF-8 bytes of the document and source_sha256 their "
            "SHA-256."
        ),
    },
    "plugins.read_nrrd": {
        "arguments": {"path": "tiny.nrrd", "offset": 0, "limit": 2},
        # NRRD0004 ASCII-encoded 3x2 int16 raster. The blank line at byte 98 ends the
        # header, so data starts at byte 99; the six decimal tokens are separated by
        # single spaces and the file ends with one newline. Written as literal UTF-8 text.
        "files": {
            "tiny.nrrd": (
                "NRRD0004\n"
                "type: int16\n"
                "dimension: 2\n"
                "sizes: 3 2\n"
                "encoding: ascii\n"
                "spacings: 0.5 nan\n"
                "content: synthetic\n"
                "\n"
                "1 -2 3 -4 5 -6\n"
            )
        },
        "expect": {
            "version": 4,
            "scalar_type": "int16",
            "item_bytes": 2,
            "encoding": "ascii",
            "endian_declaration": None,
            "decoded_byte_order": "not_applicable",
            "axes": [
                {
                    "axis_index": 0,
                    "size": 3,
                    "element_stride": 1,
                    "label": None,
                    "unit": None,
                    "spacing": "0.5",
                    "minimum": None,
                    "maximum": None,
                },
                {
                    "axis_index": 1,
                    "size": 2,
                    "element_stride": 3,
                    "label": None,
                    "unit": None,
                    "spacing": "nan",
                    "minimum": None,
                    "maximum": None,
                },
            ],
            "sample_count": 6,
            "raw_equivalent_bytes": 12,
            "data_byte_start": 99,
            "data_bytes": 15,
            "header_lines": [
                "NRRD0004",
                "type: int16",
                "dimension: 2",
                "sizes: 3 2",
                "encoding: ascii",
                "spacings: 0.5 nan",
                "content: synthetic",
                "",
            ],
            "content_declaration": "synthetic",
            "sample_units_declaration": None,
            "deprecated_number_declaration": None,
            "custom_metadata": [],
            "overwritten_custom_key_count": 0,
            "samples": [
                {
                    "sample_index": 0,
                    "coordinate": [0, 0],
                    "source_byte_start": 99,
                    "source_byte_length": 1,
                    "raw_hex": "31",
                    "value": "1",
                },
                {
                    "sample_index": 1,
                    "coordinate": [1, 0],
                    "source_byte_start": 101,
                    "source_byte_length": 2,
                    "raw_hex": "2d32",
                    "value": "-2",
                },
            ],
            "offset": 0,
            "has_more": True,
            "axis_order": "first_axis_fastest",
            "ascii_binary_quantization_applied": False,
            "full_supported_source_validated": True,
            "source_bytes": 114,
            "source_sha256": "ab31f6e9949ac2e211f161cff03cef0fdb374f72cc4548a9dced2ac890a19c75",
        },
        "approx": False,
        "note": (
            "The magic line NRRD0004 gives version 4. 'type: int16' is already a canonical "
            "type name with width 2, so item_bytes is 2. 'encoding: ascii' needs no endian "
            "declaration, hence endian_declaration null and decoded_byte_order "
            "not_applicable. Counting header bytes line by line (each line plus its LF): "
            "NRRD0004 0..8, type 9..20, dimension 21..33, sizes 34..44, encoding 45..60, "
            "spacings 61..78, content 79..97, then the blank terminator line at byte 98, so "
            "data_byte_start is 99 and header_lines has all 8 lines including that final "
            "empty one. 'sizes: 3 2' with dimension 2 gives sample_count 3 * 2 = 6 and "
            "raw_equivalent_bytes 6 * 2 = 12. The payload is the 15 bytes "
            "'1 -2 3 -4 5 -6\\n', so data_bytes is 114 - 99 = 15 and source_bytes is 114. "
            "Splitting that payload on whitespace puts the six tokens at absolute offsets 99 "
            "(len 1), 101 (len 2), 104 (len 1), 106 (len 2), 109 (len 1) and 111 (len 2). "
            "Coordinates are first-axis-fastest, so sample 0 is [0, 0] and sample 1 is "
            "[1, 0]; the requested page is offset 0 limit 2, so only those two are returned "
            "and has_more is 0 + 2 < 6 = True. Token bytes are 0x31 ('1') and 0x2d32 ('-2'), "
            "both inside the int16 range -32768..32767, and ASCII integers are reported as "
            "exact decimal strings with no float32/float64 quantization. 'spacings: 0.5 nan' "
            "yields spacing '0.5' for axis 0 (Decimal('0.5') printed back) and the literal "
            "'nan' unknown marker for axis 1; element strides are the cumulative products of "
            "the sizes, 1 then 3. No labels, units, axis mins/maxs, sample units, number or "
            "custom key:=value lines are present."
        ),
    },
    "plugins.read_stl": {
        "arguments": {"path": "tiny.stl", "format": "binary"},
        # Binary STL: an 80-byte header (b"fx1" then 77 zero bytes), uint32 triangle
        # count 1, then one 50-byte triangle of 12 little-endian float32 values followed
        # by a zero uint16 attribute word. Normal (0, 0, 1); vertices (0,0,0), (1,0,0),
        # (0,1,0). Every value is exactly representable in float32.
        "files": {
            "tiny.stl": {
                "base64": "ZngxAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAA="
            }
        },
        "expect": {
            "format": "binary",
            "value_encoding": "float_hex",
            "solid_name": None,
            "binary_header_hex": "667831" + "00" * 77,
            "triangle_count": 1,
            "has_triangles": True,
            "repeated_vertex_triangles": 0,
            "exactly_degenerate_triangles": 0,
            "exactly_zero_declared_normals": 0,
            "exactly_unit_declared_normals": 1,
            "positive_normal_winding_dot": 1,
            "negative_normal_winding_dot": 0,
            "zero_normal_winding_dot": 0,
            "coordinate_minimum": ["0", "0", "0"],
            "coordinate_maximum": ["1", "1", "0"],
            "coordinate_bound_encoding": "exact_rational",
            "geometry_arithmetic": "exact_rational_on_reported_scalar_values",
            "ascii_float32_quantization_applied": False,
            "adjacency_checked": False,
            "mesh_validity_verified": False,
            "format_inferred": False,
            "triangles": [
                {
                    "triangle_index": 0,
                    "source_byte_start": 84,
                    "source_byte_length": 50,
                    "normal": [
                        {
                            "value": "0x0.0p+0",
                            "source_byte_start": 84,
                            "source_byte_length": 4,
                            "raw_hex": "00000000",
                        },
                        {
                            "value": "0x0.0p+0",
                            "source_byte_start": 88,
                            "source_byte_length": 4,
                            "raw_hex": "00000000",
                        },
                        {
                            "value": "0x1.0000000000000p+0",
                            "source_byte_start": 92,
                            "source_byte_length": 4,
                            "raw_hex": "0000803f",
                        },
                    ],
                    "vertices": [
                        [
                            {
                                "value": "0x0.0p+0",
                                "source_byte_start": 96,
                                "source_byte_length": 4,
                                "raw_hex": "00000000",
                            },
                            {
                                "value": "0x0.0p+0",
                                "source_byte_start": 100,
                                "source_byte_length": 4,
                                "raw_hex": "00000000",
                            },
                            {
                                "value": "0x0.0p+0",
                                "source_byte_start": 104,
                                "source_byte_length": 4,
                                "raw_hex": "00000000",
                            },
                        ],
                        [
                            {
                                "value": "0x1.0000000000000p+0",
                                "source_byte_start": 108,
                                "source_byte_length": 4,
                                "raw_hex": "0000803f",
                            },
                            {
                                "value": "0x0.0p+0",
                                "source_byte_start": 112,
                                "source_byte_length": 4,
                                "raw_hex": "00000000",
                            },
                            {
                                "value": "0x0.0p+0",
                                "source_byte_start": 116,
                                "source_byte_length": 4,
                                "raw_hex": "00000000",
                            },
                        ],
                        [
                            {
                                "value": "0x0.0p+0",
                                "source_byte_start": 120,
                                "source_byte_length": 4,
                                "raw_hex": "00000000",
                            },
                            {
                                "value": "0x1.0000000000000p+0",
                                "source_byte_start": 124,
                                "source_byte_length": 4,
                                "raw_hex": "0000803f",
                            },
                            {
                                "value": "0x0.0p+0",
                                "source_byte_start": 128,
                                "source_byte_length": 4,
                                "raw_hex": "00000000",
                            },
                        ],
                    ],
                    "has_repeated_vertex": False,
                    "is_exactly_degenerate": False,
                    "declared_normal_is_exactly_zero": False,
                    "declared_normal_is_exactly_unit": True,
                    "normal_winding_dot_sign": 1,
                }
            ],
            "offset": 0,
            "has_more": False,
            "source_bytes": 134,
            "source_sha256": "01fd47ead1bf52b44ca351080aefda44f9e1415b9aed8bf2b17c0a78ba38749c",
        },
        "approx": False,
        "note": (
            "The caller declares format 'binary', so nothing is inferred and solid_name "
            "stays null while binary_header_hex is the first 80 bytes: b'fx1' is 667831 "
            "followed by 77 zero bytes (154 hex zeros), 160 hex characters in total. Bytes "
            "80..84 are the little-endian uint32 triangle count 1, so the file must be "
            "exactly 84 + 50 * 1 = 134 bytes, which is source_bytes. The single triangle "
            "occupies bytes 84..134 and bytes 132..134 are the attribute word, which must be "
            "zero. Its 12 float32 values start at byte 84 in 4-byte steps: the declared "
            "normal is (0, 0, 1) at 84/88/92 and the three vertices are (0,0,0) at 96/100/"
            "104, (1,0,0) at 108/112/116 and (0,1,0) at 120/124/128. Little-endian float32 "
            "1.0 is the bytes 0000803f (0x3F800000) and 0.0 is 00000000; read back as Python "
            "floats they are exactly 1.0 and 0.0, whose float.hex() spellings are "
            "'0x1.0000000000000p+0' and '0x0.0p+0'. Geometry in exact rationals: v1 - v0 = "
            "(1,0,0) and v2 - v0 = (0,1,0), so the cross product is (0*0-0*1, 0*0-1*0, "
            "1*1-0*0) = (0,0,1), which is nonzero, so the triangle is not exactly "
            "degenerate. The three vertices are distinct, so has_repeated_vertex is False. "
            "The normal's squared length is 0+0+1 = 1, so it is exactly unit and not exactly "
            "zero. normal dot cross = 0*0 + 0*0 + 1*1 = 1 > 0, giving "
            "normal_winding_dot_sign 1 and one positive winding observation. Coordinate "
            "bounds are taken over the vertices only: minimum (0,0,0) and maximum (1,1,0), "
            "rendered as exact rational strings '0' and '1'. The default page (offset 0, "
            "limit 50) returns the only triangle, so has_more is 0 + 1 < 1 = False."
        ),
    },
}
