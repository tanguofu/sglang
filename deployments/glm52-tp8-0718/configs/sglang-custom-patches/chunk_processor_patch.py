import re

filepath = "sgl-model-gateway/src/routers/openai/responses/common.rs"
with open(filepath, "r") as f:
    content = f.read()

# Replace the ChunkProcessor struct and impl block
old = '''/// Processes incoming byte chunks into complete SSE blocks.
/// Handles buffering of partial chunks and CRLF normalization.
pub(super) struct ChunkProcessor {
    pending: String,
}

impl ChunkProcessor {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
        }
    }

    /// Append a chunk to the buffer, normalizing line endings
    pub fn push_chunk(&mut self, chunk: &[u8]) {
        let chunk_str = match std::str::from_utf8(chunk) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(String::from_utf8_lossy(chunk).into_owned()),
        };
        // Normalize CRLF to LF without extra allocation
        let mut chars = chunk_str.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\r' && chars.peek() == Some(&'\\n') {
                // Skip \\r when followed by \\n
                continue;
            }
            self.pending.push(c);
        }
    }

    /// Extract the next complete SSE block from the buffer, if available
    pub fn next_block(&mut self) -> Option<String> {
        loop {
            let pos = self.pending.find("\\n\\n")?;
            let block = self.pending[..pos].to_string();
            self.pending.drain(..pos + 2);

            if !block.trim().is_empty() {
                return Some(block);
            }
            // If block is empty, loop again to find the next one
        }
    }

    /// Check if there's remaining content in the buffer
    pub fn has_remaining(&self) -> bool {
        !self.pending.trim().is_empty()
    }

    /// Take any remaining content from the buffer
    pub fn take_remaining(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}'''

new = '''/// Processes incoming byte chunks into complete SSE blocks.
/// Handles buffering of partial chunks and CRLF normalization.
///
/// Uses a `Vec<u8>` buffer instead of `String` to avoid corrupting
/// multi-byte UTF-8 sequences that may be split across chunk boundaries.
/// The previous `String::from_utf8_lossy` approach replaced incomplete
/// trailing bytes with U+FFFD, producing invalid JSON in SSE events.
pub(super) struct ChunkProcessor {
    pending: Vec<u8>,
}

impl ChunkProcessor {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Append a chunk to the buffer, normalizing CRLF to LF.
    /// Raw bytes are stored so that multi-byte UTF-8 sequences
    /// split across chunks are reassembled correctly.
    pub fn push_chunk(&mut self, chunk: &[u8]) {
        // Handle cross-chunk CRLF: if the buffer ends with \\r and the
        // new chunk starts with \\n, remove the trailing \\r.
        if self.pending.last() == Some(&b'\\r') && chunk.first() == Some(&b'\\n') {
            self.pending.pop();
        }

        // Normalize CRLF to LF within the chunk (skip \\r when followed by \\n)
        let mut start = 0;
        for i in 0..chunk.len() {
            if chunk[i] == b'\\r' && i + 1 < chunk.len() && chunk[i + 1] == b'\\n' {
                self.pending.extend_from_slice(&chunk[start..i]);
                start = i + 1; // skip \\r, keep \\n
            }
        }
        self.pending.extend_from_slice(&chunk[start..]);
    }

    /// Extract the next complete SSE block from the buffer, if available.
    /// The block separator (\\n\\n) is ASCII, so the byte-level search is
    /// safe regardless of UTF-8 content. Conversion to String happens only
    /// on a complete block, where all multi-byte sequences are intact.
    pub fn next_block(&mut self) -> Option<String> {
        loop {
            let pos = self.pending.windows(2).position(|w| w == b"\\n\\n")?;
            let block_bytes = &self.pending[..pos];
            let block = String::from_utf8(block_bytes.to_vec())
                .unwrap_or_else(|_| String::from_utf8_lossy(block_bytes).into_owned());
            self.pending.drain(..pos + 2);

            if !block.trim().is_empty() {
                return Some(block);
            }
        }
    }

    /// Check if there's remaining content in the buffer
    pub fn has_remaining(&self) -> bool {
        !self.pending.iter().all(|&b| b.is_ascii_whitespace())
    }

    /// Take any remaining content from the buffer
    pub fn take_remaining(&mut self) -> String {
        let bytes = std::mem::take(&mut self.pending);
        String::from_utf8(bytes)
            .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned())
    }
}'''

if old not in content:
    print("ERROR: old text not found in file!")
    import sys
    sys.exit(1)

content = content.replace(old, new)

with open(filepath, "w") as f:
    f.write(content)

print("Patch applied successfully!")
