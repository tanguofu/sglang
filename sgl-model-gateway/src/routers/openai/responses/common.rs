//! Common SSE parsing and processing utilities for OpenAI responses
//!
//! This module contains shared helpers used by both streaming and accumulator modules.

use std::borrow::Cow;

use serde_json::Value;

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract output_index from a JSON value
#[inline]
pub(super) fn extract_output_index(value: &Value) -> Option<usize> {
    value.get("output_index")?.as_u64().map(|v| v as usize)
}

/// Get event type from event name or parsed JSON, returning a reference to avoid allocation
#[inline]
pub(super) fn get_event_type<'a>(event_name: Option<&'a str>, parsed: &'a Value) -> &'a str {
    event_name
        .or_else(|| parsed.get("type").and_then(|v| v.as_str()))
        .unwrap_or("")
}

// ============================================================================
// Chunk Processor
// ============================================================================

/// Processes incoming byte chunks into complete SSE blocks.
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
        // Handle cross-chunk CRLF: if the buffer ends with \r and the
        // new chunk starts with \n, remove the trailing \r.
        if self.pending.last() == Some(&b'\r') && chunk.first() == Some(&b'\n') {
            self.pending.pop();
        }

        // Normalize CRLF to LF within the chunk (skip \r when followed by \n)
        let mut start = 0;
        for i in 0..chunk.len() {
            if chunk[i] == b'\r' && i + 1 < chunk.len() && chunk[i + 1] == b'\n' {
                self.pending.extend_from_slice(&chunk[start..i]);
                start = i + 1; // skip \r, keep \n
            }
        }
        self.pending.extend_from_slice(&chunk[start..]);
    }

    /// Extract the next complete SSE block from the buffer, if available.
    /// The block separator (\n\n) is ASCII, so the byte-level search is
    /// safe regardless of UTF-8 content. Conversion to String happens only
    /// on a complete block, where all multi-byte sequences are intact.
    pub fn next_block(&mut self) -> Option<String> {
        loop {
            let pos = self.pending.windows(2).position(|w| w == b"\n\n")?;
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
        match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        }
    }
}

// ============================================================================
// SSE Parsing
// ============================================================================

/// Parse an SSE block into event name and data
///
/// Returns borrowed strings when possible to avoid allocations in hot paths.
/// Only allocates when multiple data lines need to be joined.
pub(super) fn parse_sse_block(block: &str) -> (Option<&str>, Cow<'_, str>) {
    let mut event_name: Option<&str> = None;
    let mut data_lines: Vec<&str> = Vec::new();

    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }

    let data = if data_lines.len() == 1 {
        Cow::Borrowed(data_lines[0])
    } else {
        Cow::Owned(data_lines.join("\n"))
    };

    (event_name, data)
}
