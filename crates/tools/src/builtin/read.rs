use crate::{Tool, ToolError};
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::AsyncReadExt;

const DEFAULT_LIMIT: usize = 2000;
const MAX_BYTES: usize = 512 * 1024; // 512 KB
const BINARY_SAMPLE_BYTES: usize = 8192;

/// Reads a file and returns its contents with line numbers.
///
/// Supports pagination via `offset` (1-indexed start line) and `limit` (max lines).
/// Binary files are rejected with a descriptive error.
/// When output is truncated a `continuation` hint is included in the response.
#[derive(Clone)]
pub struct ReadTool;

impl ReadTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Reads a file and returns its contents with line numbers. \
        Use offset (1-indexed) and limit to paginate large files. \
        Returns an error for directories and binary files. \
        When the file is truncated, a continuation hint tells you the next offset to use."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read."
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed line number to start reading from. Defaults to 1.",
                    "minimum": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Defaults to 2000.",
                    "minimum": 1
                }
            },
            "required": ["file_path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let file_path = input
            .get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput {
                name: self.name().to_string(),
                reason: "missing or invalid 'file_path' field".to_string(),
            })?;

        let offset = input
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_LIMIT);

        if offset == 0 {
            return Err(ToolError::InvalidInput {
                name: self.name().to_string(),
                reason: "offset must be >= 1".to_string(),
            });
        }

        // Stat the file
        let metadata =
            tokio::fs::metadata(file_path)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    name: self.name().to_string(),
                    reason: format!("cannot access '{}': {}", file_path, e),
                })?;

        if metadata.is_dir() {
            return Err(ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("'{}' is a directory, not a file", file_path),
            });
        }

        // Read raw bytes for binary detection
        let mut file =
            tokio::fs::File::open(file_path)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    name: self.name().to_string(),
                    reason: format!("cannot open '{}': {}", file_path, e),
                })?;

        let sample_size = BINARY_SAMPLE_BYTES.min(metadata.len() as usize);
        let mut sample = vec![0u8; sample_size];
        file.read_exact(&mut sample)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("cannot read '{}': {}", file_path, e),
            })?;

        if is_binary(&sample) {
            return Err(ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!(
                    "'{}' appears to be a binary file and cannot be read as text",
                    file_path
                ),
            });
        }

        // Read full file as text
        let contents =
            tokio::fs::read_to_string(file_path)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    name: self.name().to_string(),
                    reason: format!("cannot read '{}': {}", file_path, e),
                })?;

        let all_lines: Vec<&str> = contents.lines().collect();
        let total_lines = all_lines.len();

        // offset is 1-indexed; clamp to valid range
        let start = (offset - 1).min(total_lines);
        let end = (start + limit).min(total_lines);

        let mut output = String::new();
        let mut byte_count = 0usize;
        let mut actual_end = start;

        for (i, line) in all_lines[start..end].iter().enumerate() {
            let line_num = start + i + 1; // 1-indexed
            let entry = format!("{}: {}\n", line_num, line);
            byte_count += entry.len();

            if byte_count > MAX_BYTES {
                break;
            }

            output.push_str(&entry);
            actual_end = start + i + 1;
        }

        // If nothing was written, the first line alone exceeds MAX_BYTES.
        // Include it in full anyway to guarantee the caller always makes forward progress.
        if actual_end == start && start < total_lines {
            let line_num = start + 1;
            output = format!("{}: {}\n", line_num, all_lines[start]);
            actual_end = start + 1;
        }

        // Ensure from <= to: when offset is past EOF nothing is read (actual_end == start)
        // and we use (0, 0) to signal "no lines returned" rather than emitting an invalid range.
        let (from_line, to_line) = if total_lines == 0 || actual_end == start {
            (0, 0)
        } else {
            (start + 1, actual_end)
        };
        let is_truncated = actual_end < total_lines;

        let mut result = serde_json::json!({
            "path": file_path,
            "type": "file",
            "lines": {
                "from": from_line,
                "to": to_line,
                "total": total_lines
            },
            "content": output.trim_end_matches('\n')
        });

        if is_truncated {
            result["continuation"] = Value::String(format!(
                "Showing lines {}-{} of {}. Use offset={} to continue reading.",
                from_line,
                to_line,
                total_lines,
                actual_end + 1
            ));
        }

        Ok(result)
    }
}

/// Detects binary content by scanning for null bytes.
fn is_binary(sample: &[u8]) -> bool {
    sample.contains(&0u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_input(path: &str, offset: Option<u64>, limit: Option<u64>) -> Value {
        let mut map = serde_json::json!({ "file_path": path });
        if let Some(o) = offset {
            map["offset"] = Value::Number(o.into());
        }
        if let Some(l) = limit {
            map["limit"] = Value::Number(l.into());
        }
        map
    }

    #[tokio::test]
    async fn reads_simple_file() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "hello").unwrap();
        writeln!(f, "world").unwrap();

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), None, None))
            .await
            .unwrap();

        assert_eq!(result["type"], "file");
        assert_eq!(result["lines"]["from"], 1);
        assert_eq!(result["lines"]["to"], 2);
        assert_eq!(result["lines"]["total"], 2);
        assert!(result["content"].as_str().unwrap().contains("1: hello"));
        assert!(result["content"].as_str().unwrap().contains("2: world"));
        assert!(result.get("continuation").is_none());
    }

    #[tokio::test]
    async fn paginates_with_offset_and_limit() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), Some(4), Some(3)))
            .await
            .unwrap();

        assert_eq!(result["lines"]["from"], 4);
        assert_eq!(result["lines"]["to"], 6);
        assert_eq!(result["lines"]["total"], 10);
        let content = result["content"].as_str().unwrap();
        assert!(content.contains("4: line 4"));
        assert!(content.contains("6: line 6"));
        assert!(!content.contains("7:"));
    }

    #[tokio::test]
    async fn includes_continuation_when_truncated() {
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), Some(1), Some(3)))
            .await
            .unwrap();

        assert_eq!(result["lines"]["to"], 3);
        assert_eq!(result["lines"]["total"], 10);
        let cont = result["continuation"].as_str().unwrap();
        assert!(cont.contains("offset=4"));
    }

    #[tokio::test]
    async fn no_continuation_when_fully_read() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "only line").unwrap();

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), None, None))
            .await
            .unwrap();

        assert!(result.get("continuation").is_none());
    }

    #[tokio::test]
    async fn errors_on_missing_file() {
        let tool = ReadTool::new();
        let result = tool
            .execute(make_input("/nonexistent/path/file.rs", None, None))
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed { .. })));
    }

    #[tokio::test]
    async fn errors_on_directory() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(dir.path().to_str().unwrap(), None, None))
            .await;

        match result {
            Err(ToolError::ExecutionFailed { reason, .. }) => {
                assert!(reason.contains("is a directory"));
            }
            _ => panic!("expected ExecutionFailed for directory"),
        }
    }

    #[tokio::test]
    async fn errors_on_binary_file() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0x00, 0x01, 0x02, 0xFF, 0xFE]).unwrap();

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), None, None))
            .await;

        match result {
            Err(ToolError::ExecutionFailed { reason, .. }) => {
                assert!(reason.contains("binary file"));
            }
            _ => panic!("expected ExecutionFailed for binary file"),
        }
    }

    #[tokio::test]
    async fn errors_on_zero_offset() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line").unwrap();

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), Some(0), None))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn offset_past_eof_returns_empty_content() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "line 1").unwrap();

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), Some(999), None))
            .await
            .unwrap();

        // offset past EOF: no lines returned; from and to are both 0 (valid inclusive range)
        assert_eq!(result["lines"]["from"], 0);
        assert_eq!(result["lines"]["to"], 0);
        assert_eq!(result["lines"]["total"], 1);
        assert_eq!(result["content"].as_str().unwrap(), "");
        assert!(result.get("continuation").is_none());
    }

    #[tokio::test]
    async fn offset_past_eof_from_never_exceeds_to() {
        // Regression test: from must always be <= to regardless of offset value.
        let mut f = NamedTempFile::new().unwrap();
        for i in 1..=5 {
            writeln!(f, "line {i}").unwrap();
        }

        let tool = ReadTool::new();
        for out_of_range_offset in [6u64, 7, 100, 9999] {
            let result = tool
                .execute(make_input(
                    f.path().to_str().unwrap(),
                    Some(out_of_range_offset),
                    None,
                ))
                .await
                .unwrap();

            let from = result["lines"]["from"].as_u64().unwrap();
            let to = result["lines"]["to"].as_u64().unwrap();
            assert!(
                from <= to,
                "offset={out_of_range_offset}: expected from ({from}) <= to ({to})"
            );
        }
    }

    #[tokio::test]
    async fn long_line_returned_in_full_with_no_content_loss() {
        // A single line larger than MAX_BYTES must still be returned completely via the
        // fallback path — no silent truncation, no panic.
        let mut f = NamedTempFile::new().unwrap();
        let big = "a".repeat(MAX_BYTES + 1024);
        writeln!(f, "{big}").unwrap();

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), None, None))
            .await
            .unwrap();

        assert_eq!(result["lines"]["from"], 1);
        assert_eq!(result["lines"]["to"], 1);
        assert!(result.get("continuation").is_none());

        let content = result["content"].as_str().unwrap();
        let payload = content.trim_start_matches("1: ");
        assert_eq!(
            payload.len(),
            big.len(),
            "line content must be returned in full without any truncation"
        );
    }

    #[tokio::test]
    async fn first_line_larger_than_max_bytes_followed_by_normal_lines() {
        // Line 1 alone exceeds MAX_BYTES. The fallback forces it into the response in full.
        // Lines 2 and 3 are then NOT included (the page is full after line 1).
        // A continuation hint must be present pointing at line 2.
        let mut f = NamedTempFile::new().unwrap();
        let big = "b".repeat(MAX_BYTES + 1024);
        writeln!(f, "{big}").unwrap(); // line 1 — oversized
        writeln!(f, "line two").unwrap(); // line 2
        writeln!(f, "line three").unwrap(); // line 3

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), None, None))
            .await
            .unwrap();

        assert_eq!(result["lines"]["from"], 1);
        assert_eq!(result["lines"]["to"], 1);
        assert_eq!(result["lines"]["total"], 3);

        let cont = result["continuation"].as_str().unwrap();
        assert!(cont.contains("offset=2"), "continuation must point at line 2");

        let content = result["content"].as_str().unwrap();
        assert!(!content.contains("2:"), "line 2 must not appear in this page");

        let payload = content.trim_start_matches("1: ");
        assert_eq!(payload.len(), big.len(), "line 1 must be returned in full");
    }

    #[tokio::test]
    async fn unicode_content_is_returned_unmodified() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "héllo wörld 日本語").unwrap();
        writeln!(f, "second line").unwrap();

        let tool = ReadTool::new();
        let result = tool
            .execute(make_input(f.path().to_str().unwrap(), None, None))
            .await
            .unwrap();

        let content = result["content"].as_str().unwrap();
        assert!(content.contains("1: héllo wörld 日本語"));
        assert!(content.contains("2: second line"));
    }
}
