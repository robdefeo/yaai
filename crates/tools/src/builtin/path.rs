use crate::ToolError;
use std::path::{Path, PathBuf};

/// Resolves `input_path` and `working_dir` to canonical paths, then verifies
/// that `input_path` falls within `working_dir`. Returns `(canonical_input, canonical_wd)`.
pub(super) async fn resolve_and_check(
    working_dir: &Path,
    input_path: &Path,
    tool_name: &str,
) -> Result<(PathBuf, PathBuf), ToolError> {
    let canonical_input =
        tokio::fs::canonicalize(input_path)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                name: tool_name.to_string(),
                reason: format!("cannot access '{}': {e}", input_path.display()),
            })?;

    let canonical_wd =
        tokio::fs::canonicalize(working_dir)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                name: tool_name.to_string(),
                reason: format!("cannot canonicalize working directory: {e}"),
            })?;

    if !canonical_input.starts_with(&canonical_wd) {
        return Err(ToolError::ExecutionFailed {
            name: tool_name.to_string(),
            reason: format!(
                "'{}' is outside the working directory",
                input_path.display()
            ),
        });
    }

    Ok((canonical_input, canonical_wd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn allows_path_within_working_dir() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("file.txt");
        tokio::fs::write(&file, b"").await.unwrap();

        let result = resolve_and_check(dir.path(), &file, "tool").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn allows_working_dir_itself() {
        let dir = tempdir().unwrap();
        let result = resolve_and_check(dir.path(), dir.path(), "tool").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn rejects_path_outside_working_dir() {
        let inside = tempdir().unwrap();
        let outside = tempdir().unwrap();

        let err = resolve_and_check(inside.path(), outside.path(), "tool")
            .await
            .unwrap_err();
        match err {
            ToolError::ExecutionFailed { reason, .. } => {
                assert!(reason.contains("outside the working directory"));
            }
            _ => panic!("expected ExecutionFailed"),
        }
    }

    #[tokio::test]
    async fn rejects_nonexistent_path() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("does_not_exist.txt");

        let err = resolve_and_check(dir.path(), &missing, "tool")
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed { .. }));
    }
}
