//! Shared SSE stream utilities.

use anyhow::{Context, Result};

/// Pop one complete line from a byte buffer of SSE data.
///
/// Returns `None` when no newline has arrived yet (caller should buffer more
/// bytes). Strips the trailing `\r\n` or `\n`.
pub(crate) fn pop_sse_line(buf: &mut Vec<u8>) -> Result<Option<String>> {
    let Some(pos) = buf.iter().position(|byte| *byte == b'\n') else {
        return Ok(None);
    };

    let mut line: Vec<u8> = buf.drain(..=pos).collect();
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }

    String::from_utf8(line)
        .context("decoding SSE line")
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_line_buffer_preserves_utf8_split_across_chunks() {
        let line = "data: hello 😀\n";
        let split = line.find('😀').unwrap() + 1;
        let bytes = line.as_bytes();
        let mut buf = Vec::new();

        buf.extend_from_slice(&bytes[..split]);
        assert_eq!(pop_sse_line(&mut buf).unwrap(), None);

        buf.extend_from_slice(&bytes[split..]);
        assert_eq!(
            pop_sse_line(&mut buf).unwrap().as_deref(),
            Some(line.trim_end_matches('\n'))
        );
    }
}
