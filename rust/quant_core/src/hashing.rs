//! SHA-256 hex digests. Output matches `hashlib.sha256(...).hexdigest()`.

use sha2::{Digest, Sha256};

const HEX: &[u8; 16] = b"0123456789abcdef";

/// ASCII hex of the SHA-256 digest. Always 64 lowercase bytes.
pub fn sha256_hex_ascii(data: &[u8]) -> [u8; 64] {
    let digest = Sha256::digest(data);
    let mut out = [0u8; 64];
    for (i, &byte) in digest.iter().enumerate() {
        out[i * 2] = HEX[(byte >> 4) as usize];
        out[i * 2 + 1] = HEX[(byte & 0x0f) as usize];
    }
    out
}

pub fn sha256_hex(data: &[u8]) -> String {
    let hex = sha256_hex_ascii(data);
    // Hex digits are ASCII, so this conversion cannot fail.
    String::from_utf8(hex.to_vec()).expect("sha256 hex is ascii")
}

/// Digest a packed buffer of blobs. `ends` are exclusive offsets into `buf`.
pub fn hash_packed(buf: &[u8], ends: &[usize]) -> Vec<[u8; 64]> {
    let n = ends.len();
    let mut out = vec![[0u8; 64]; n];
    if n == 0 {
        return out;
    }
    // Short batches are cheaper on one core. The native bench is 20k small
    // blobs, where the per-call GIL tax otherwise loses to CPython 3.13.
    let workers = if n < 1024 {
        1
    } else {
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 8)
    };
    if workers == 1 {
        fill_range(&mut out, buf, ends, 0);
        return out;
    }
    let chunk = n.div_ceil(workers);
    std::thread::scope(|scope| {
        for (index, slot) in out.chunks_mut(chunk).enumerate() {
            let start_index = index * chunk;
            scope.spawn(move || fill_range(slot, buf, ends, start_index));
        }
    });
    out
}

fn fill_range(slot: &mut [[u8; 64]], buf: &[u8], ends: &[usize], start_index: usize) {
    let mut start = if start_index == 0 {
        0
    } else {
        ends[start_index - 1]
    };
    for (offset, hex) in slot.iter_mut().enumerate() {
        let end = ends[start_index + offset];
        *hex = sha256_hex_ascii(&buf[start..end]);
        start = end;
    }
}
