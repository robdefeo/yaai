use crate::{Tool, ToolError};
use async_trait::async_trait;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

const DEFAULT_DEPTH: u32 = 2;
const DEFAULT_LIMIT: u64 = 25;
const MAX_BFS_ENTRIES: usize = 10_000;

#[derive(Debug, Deserialize, JsonSchema)]
struct ListDirInput {
    /// Absolute path to the directory to list.
    dir_path: String,
    /// Maximum traversal depth. Minimum 1. Defaults to 2.
    depth: Option<u32>,
    /// Maximum number of entries to return. Defaults to 25.
    limit: Option<u64>,
    /// 1-indexed entry number to start from, for pagination. Defaults to 1.
    offset: Option<u64>,
    /// Whether to include hidden entries (names starting with '.'). Defaults to false.
    include_hidden: Option<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    File,
    Dir,
    Symlink,
}

struct BfsEntry {
    name: String,
    kind: EntryKind,
    depth: usize,
}

#[derive(Serialize)]
struct ListDirResult {
    entries: String,
    total: usize,
    truncated: bool,
}

/// Lists directory contents as an indented tree, bounded by depth and paginated by limit/offset.
#[derive(Clone)]
pub struct ListDirTool {
    working_dir: Option<PathBuf>,
}

impl ListDirTool {
    pub fn new() -> Self {
        Self { working_dir: None }
    }

    async fn resolve_working_dir(&self) -> Result<PathBuf, ToolError> {
        let raw = match &self.working_dir {
            Some(p) => p.clone(),
            None => std::env::current_dir().map_err(|e| ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("cannot determine working directory: {e}"),
            })?,
        };
        tokio::fs::canonicalize(&raw)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("cannot canonicalize working directory: {e}"),
            })
    }
}

impl Default for ListDirTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "Lists the contents of a directory as a tree. Use to discover files before reading or editing them."
    }

    fn input_schema(&self) -> Value {
        let mut schema = serde_json::to_value(schema_for!(ListDirInput))
            .expect("ListDirInput schema is always valid");
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$schema");
            obj.remove("title");
        }
        schema
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let params: ListDirInput =
            serde_json::from_value(input).map_err(|e| ToolError::InvalidInput {
                name: self.name().to_string(),
                reason: e.to_string(),
            })?;

        let max_depth = params.depth.unwrap_or(DEFAULT_DEPTH) as usize;
        let limit = params.limit.unwrap_or(DEFAULT_LIMIT) as usize;
        let offset = params.offset.unwrap_or(1) as usize;
        let include_hidden = params.include_hidden.unwrap_or(false);

        if max_depth == 0 {
            return Err(ToolError::InvalidInput {
                name: self.name().to_string(),
                reason: "depth must be >= 1".to_string(),
            });
        }
        if offset == 0 {
            return Err(ToolError::InvalidInput {
                name: self.name().to_string(),
                reason: "offset must be >= 1".to_string(),
            });
        }
        if limit == 0 {
            return Err(ToolError::InvalidInput {
                name: self.name().to_string(),
                reason: "limit must be >= 1".to_string(),
            });
        }

        let canonical_target = tokio::fs::canonicalize(&params.dir_path)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("cannot access '{}': {e}", params.dir_path),
            })?;

        let canonical_cwd = self.resolve_working_dir().await?;

        if !canonical_target.starts_with(&canonical_cwd) {
            return Err(ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("'{}' is outside the working directory", params.dir_path),
            });
        }

        let metadata = tokio::fs::metadata(&canonical_target).await.map_err(|e| {
            ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("cannot stat '{}': {e}", params.dir_path),
            }
        })?;

        if !metadata.is_dir() {
            return Err(ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: format!("'{}' is not a directory", params.dir_path),
            });
        }

        let all_entries = collect_entries(&canonical_target, max_depth, include_hidden).await;
        let total = all_entries.len();
        let start = (offset - 1).min(total);
        let end = start.saturating_add(limit).min(total);
        let page = &all_entries[start..end];
        let truncated = (offset - 1).saturating_add(limit) < total;

        let mut lines = vec![format!("{}/", canonical_target.display())];
        for entry in page {
            let indent = "  ".repeat(entry.depth);
            let suffix = match entry.kind {
                EntryKind::Dir => "/",
                EntryKind::Symlink => "@",
                EntryKind::File => "",
            };
            lines.push(format!("{}{}{}", indent, entry.name, suffix));
        }

        let result = ListDirResult {
            entries: lines.join("\n"),
            total,
            truncated,
        };

        Ok(serde_json::to_value(result).expect("ListDirResult is always serializable"))
    }
}

/// BFS traversal of `root` up to `max_depth` levels deep.
/// Skips unreadable entries and does not follow symlinks.
/// Sorts children lexicographically at each level before enqueuing.
async fn collect_entries(root: &Path, max_depth: usize, include_hidden: bool) -> Vec<BfsEntry> {
    let mut entries = Vec::new();
    // queue: (directory path, depth of that directory, relative to root)
    // root itself is at depth 0; its children are at depth 1.
    let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::new();
    queue.push_back((root.to_path_buf(), 0));

    while let Some((dir, parent_depth)) = queue.pop_front() {
        let child_depth = parent_depth + 1;

        let mut read_dir = match tokio::fs::read_dir(&dir).await {
            Ok(rd) => rd,
            Err(_) => continue,
        };

        let mut children: Vec<(String, EntryKind, PathBuf)> = Vec::new();

        while let Ok(Some(de)) = read_dir.next_entry().await {
            let name = de.file_name().to_string_lossy().into_owned();
            if !include_hidden && name.starts_with('.') {
                continue;
            }
            let path = de.path();
            let meta = match tokio::fs::symlink_metadata(&path).await {
                Ok(m) => m,
                Err(_) => continue,
            };
            let kind = if meta.file_type().is_symlink() {
                EntryKind::Symlink
            } else if meta.is_dir() {
                EntryKind::Dir
            } else {
                EntryKind::File
            };
            children.push((name, kind, path));
        }

        children.sort_by(|a, b| a.0.cmp(&b.0));

        for (name, kind, path) in children {
            entries.push(BfsEntry {
                name,
                kind,
                depth: child_depth,
            });
            if entries.len() >= MAX_BFS_ENTRIES {
                return entries;
            }
            if kind == EntryKind::Dir && child_depth < max_depth {
                queue.push_back((path, child_depth));
            }
        }
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    impl ListDirTool {
        fn with_working_dir(dir: impl Into<PathBuf>) -> Self {
            Self {
                working_dir: Some(dir.into()),
            }
        }
    }

    #[tokio::test]
    async fn lists_files_and_dirs() {
        let root = tempdir().unwrap();
        tokio::fs::write(root.path().join("a.txt"), b"")
            .await
            .unwrap();
        tokio::fs::write(root.path().join("b.txt"), b"")
            .await
            .unwrap();
        tokio::fs::create_dir(root.path().join("src"))
            .await
            .unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap() }))
            .await
            .unwrap();

        let entries = result["entries"].as_str().unwrap();
        assert!(entries.contains("a.txt"));
        assert!(entries.contains("b.txt"));
        assert!(entries.contains("src/"));
        assert_eq!(result["total"], 3);
        assert_eq!(result["truncated"], false);
    }

    #[tokio::test]
    async fn sorts_entries_lexicographically() {
        let root = tempdir().unwrap();
        tokio::fs::write(root.path().join("z.txt"), b"")
            .await
            .unwrap();
        tokio::fs::write(root.path().join("a.txt"), b"")
            .await
            .unwrap();
        tokio::fs::write(root.path().join("m.txt"), b"")
            .await
            .unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap() }))
            .await
            .unwrap();

        let entries = result["entries"].as_str().unwrap();
        let pos_a = entries.find("a.txt").unwrap();
        let pos_m = entries.find("m.txt").unwrap();
        let pos_z = entries.find("z.txt").unwrap();
        assert!(pos_a < pos_m && pos_m < pos_z);
    }

    #[tokio::test]
    async fn respects_depth_limit() {
        let root = tempdir().unwrap();
        let sub = root.path().join("sub");
        tokio::fs::create_dir(&sub).await.unwrap();
        tokio::fs::write(sub.join("deep.txt"), b"").await.unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap(), "depth": 1 }))
            .await
            .unwrap();

        let entries = result["entries"].as_str().unwrap();
        assert!(entries.contains("sub/"));
        assert!(
            !entries.contains("deep.txt"),
            "depth=1 must not show grandchildren"
        );
    }

    #[tokio::test]
    async fn default_depth_shows_two_levels() {
        let root = tempdir().unwrap();
        let sub = root.path().join("sub");
        tokio::fs::create_dir(&sub).await.unwrap();
        tokio::fs::write(sub.join("deep.txt"), b"").await.unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap() }))
            .await
            .unwrap();

        let entries = result["entries"].as_str().unwrap();
        assert!(entries.contains("sub/"));
        assert!(
            entries.contains("deep.txt"),
            "default depth=2 must show grandchildren"
        );
    }

    #[tokio::test]
    async fn paginates_with_offset_and_limit() {
        let root = tempdir().unwrap();
        for i in 0..10u8 {
            tokio::fs::write(root.path().join(format!("{i:02}.txt")), b"")
                .await
                .unwrap();
        }

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({
                "dir_path": root.path().to_str().unwrap(),
                "offset": 3,
                "limit": 2
            }))
            .await
            .unwrap();

        // offset=3, limit=2 → indices 2..4 → "02.txt", "03.txt"
        assert_eq!(result["total"], 10);
        let entries = result["entries"].as_str().unwrap();
        assert!(entries.contains("02.txt"));
        assert!(entries.contains("03.txt"));
        assert!(!entries.contains("00.txt"));
        assert!(!entries.contains("04.txt"));
    }

    #[tokio::test]
    async fn reports_truncated_when_more_entries_exist() {
        let root = tempdir().unwrap();
        for i in 0..10u8 {
            tokio::fs::write(root.path().join(format!("{i}.txt")), b"")
                .await
                .unwrap();
        }

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap(), "limit": 3 }))
            .await
            .unwrap();

        assert_eq!(result["total"], 10);
        assert_eq!(result["truncated"], true);
    }

    #[tokio::test]
    async fn not_truncated_when_all_entries_fit() {
        let root = tempdir().unwrap();
        tokio::fs::write(root.path().join("a.txt"), b"")
            .await
            .unwrap();
        tokio::fs::write(root.path().join("b.txt"), b"")
            .await
            .unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap() }))
            .await
            .unwrap();

        assert_eq!(result["truncated"], false);
    }

    #[tokio::test]
    async fn errors_on_nonexistent_path() {
        let root = tempdir().unwrap();
        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({
                "dir_path": root.path().join("no_such_dir").to_str().unwrap()
            }))
            .await;

        assert!(matches!(result, Err(ToolError::ExecutionFailed { .. })));
    }

    #[tokio::test]
    async fn errors_on_file_path() {
        let root = tempdir().unwrap();
        let file = root.path().join("file.txt");
        tokio::fs::write(&file, b"content").await.unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": file.to_str().unwrap() }))
            .await;

        match result {
            Err(ToolError::ExecutionFailed { reason, .. }) => {
                assert!(reason.contains("not a directory"));
            }
            _ => panic!("expected ExecutionFailed for file path"),
        }
    }

    #[tokio::test]
    async fn errors_on_path_outside_working_directory() {
        let root = tempdir().unwrap();
        let outside = tempdir().unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": outside.path().to_str().unwrap() }))
            .await;

        match result {
            Err(ToolError::ExecutionFailed { reason, .. }) => {
                assert!(reason.contains("outside the working directory"));
            }
            _ => panic!("expected ExecutionFailed for path outside working directory"),
        }
    }

    #[tokio::test]
    async fn errors_on_zero_depth() {
        let root = tempdir().unwrap();
        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap(), "depth": 0 }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn marks_symlinks_with_at_suffix() {
        let root = tempdir().unwrap();
        tokio::fs::write(root.path().join("target.txt"), b"")
            .await
            .unwrap();
        tokio::fs::symlink(root.path().join("target.txt"), root.path().join("link.txt"))
            .await
            .unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap() }))
            .await
            .unwrap();

        let entries = result["entries"].as_str().unwrap();
        assert!(entries.contains("link.txt@"), "symlink must have @ suffix");
        assert!(entries.contains("target.txt"), "regular file has no suffix");
        assert!(!entries.contains("target.txt@"));
    }

    #[tokio::test]
    async fn errors_on_zero_offset() {
        let root = tempdir().unwrap();
        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap(), "offset": 0 }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn errors_on_zero_limit() {
        let root = tempdir().unwrap();
        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap(), "limit": 0 }))
            .await;

        assert!(matches!(result, Err(ToolError::InvalidInput { .. })));
    }

    #[tokio::test]
    async fn hides_dotfiles_by_default() {
        let root = tempdir().unwrap();
        tokio::fs::write(root.path().join(".hidden"), b"")
            .await
            .unwrap();
        tokio::fs::write(root.path().join("visible.txt"), b"")
            .await
            .unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({ "dir_path": root.path().to_str().unwrap() }))
            .await
            .unwrap();

        let entries = result["entries"].as_str().unwrap();
        assert!(entries.contains("visible.txt"));
        assert!(
            !entries.contains(".hidden"),
            "hidden entries must be excluded by default"
        );
        assert_eq!(result["total"], 1);
    }

    #[tokio::test]
    async fn shows_dotfiles_when_include_hidden_true() {
        let root = tempdir().unwrap();
        tokio::fs::write(root.path().join(".hidden"), b"")
            .await
            .unwrap();
        tokio::fs::write(root.path().join("visible.txt"), b"")
            .await
            .unwrap();

        let tool = ListDirTool::with_working_dir(root.path());
        let result = tool
            .execute(json!({
                "dir_path": root.path().to_str().unwrap(),
                "include_hidden": true
            }))
            .await
            .unwrap();

        let entries = result["entries"].as_str().unwrap();
        assert!(entries.contains("visible.txt"));
        assert!(entries.contains(".hidden"));
        assert_eq!(result["total"], 2);
    }

    #[test]
    fn input_schema_strips_draft_metadata() {
        let schema = ListDirTool::new().input_schema();
        let obj = schema.as_object().expect("schema must be an object");
        assert!(!obj.contains_key("$schema"), "$schema must be stripped");
        assert!(!obj.contains_key("title"), "title must be stripped");
        assert!(
            obj.contains_key("properties"),
            "properties must be preserved"
        );
    }
}
