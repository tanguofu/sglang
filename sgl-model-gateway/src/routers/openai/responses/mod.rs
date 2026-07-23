//! OpenAI-compatible responses handling module
//!
//! This module provides comprehensive support for OpenAI Responses API with:
//! - Streaming and non-streaming response handling
//! - MCP (Model Context Protocol) tool interception and execution
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Response accumulation for persistence
//! - Tool call detection and output index remapping

mod accumulator;
mod common;
mod mcp;
mod non_streaming;
mod streaming;
mod tool_handler;
mod utils;

// Re-export payload transformation helpers for use by context.rs and other
// modules within the crate. These functions normalize the Responses API
// payload before forwarding to SGLang workers.
pub(crate) use utils::{ensure_stream_default, unwrap_namespace_tools};

pub use non_streaming::handle_non_streaming_response;
pub use streaming::handle_streaming_response;
