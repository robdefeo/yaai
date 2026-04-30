use crate::{Tool, ToolError};
use async_trait::async_trait;
use grep_regex::RegexMatcher;
use grep_searcher::{SearcherBuilder, Sink, SinkMatch};
use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;
use schemars::{schema_for, JsonSchema};
use serde::Deserialize;
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const DEFAULT_LIMIT: usize = 50;

#[derive(Debug, Deserialize, JsonSchema)]
struct GrepFilesInput {
    /// Regular expression to match against file contents.
    pattern: String,
    /// Directory or file to search. Defaults to session working directory.
    path: Option<String>,
    /// Glob to restrict which files are searched (e.g. "*.rs", "*.{ts,tsx}").
    include: Option<String>,
    /// Maximum number of file paths to return. Defaults to 50.
    limit: Option<usize>,
}

#[derive(Clone)]
pub struct GrepFilesTool {
    working_dir: PathBuf,
}

impl GrepFilesTool {
    pub fn new() -> Self {
        Self::with_working_dir(std::env::current_dir().unwrap_or_default())
    }

    pub fn with_working_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            working_dir: dir.into(),
        }
    }
}

impl Default for GrepFilesTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GrepFilesTool {
    fn name(&self) -> &str {
        "grep_files"
    }

    fn description(&self) -> &str {
        "Finds files whose contents match a regex pattern. Returns file paths sorted by most \
        recently modified. Use to locate symbol definitions, usages, or any text across the codebase."
    }

    fn input_schema(&self) -> Value {
        let mut schema = serde_json::to_value(schema_for!(GrepFilesInput))
            .expect("GrepFilesInput schema is always valid");
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$schema");
            obj.remove("title");
        }
        schema
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let params: GrepFilesInput =
            serde_json::from_value(input).map_err(|e| ToolError::InvalidInput {
                name: self.name().to_string(),
                reason: e.to_string(),
            })?;

        let limit = params.limit.unwrap_or(DEFAULT_LIMIT);

        let search_path = params
            .path
            .as_deref()
            .map(|p| self.working_dir.join(p))
            .unwrap_or_else(|| self.working_dir.clone());

        let canonical_search =
            search_path
                .canonicalize()
                .map_err(|e| ToolError::ExecutionFailed {
                    name: self.name().to_string(),
                    reason: format!("cannot resolve path '{}': {}", search_path.display(), e),
                })?;

        let canonical_wd =
            self.working_dir
                .canonicalize()
                .map_err(|e| ToolError::ExecutionFailed {
                    name: self.name().to_string(),
                    reason: format!("cannot resolve working directory: {}", e),
                })?;

        if !canonical_search.starts_with(&canonical_wd) {
            return Err(ToolError::ExecutionFailed {
                name: self.name().to_string(),
                reason: "path outside working directory".to_string(),
            });
        }

        let matcher = RegexMatcher::new(&params.pattern).map_err(|e| ToolError::InvalidInput {
            name: self.name().to_string(),
            reason: format!("invalid regex: {}", e),
        })?;

        let include = params.include;
        let tool_name = self.name().to_string();

        let value = tokio::task::spawn_blocking(move || {
            search_files(
                &canonical_search,
                &canonical_wd,
                &matcher,
                include.as_deref(),
                limit,
                &tool_name,
            )
        })
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            name: self.name().to_string(),
            reason: format!("search task panicked: {}", e),
        })??;

        Ok(value)
    }
}

fn search_files(
    search_path: &Path,
    working_dir: &Path,
    matcher: &RegexMatcher,
    include: Option<&str>,
    limit: usize,
    tool_name: &str,
) -> Result<Value, ToolError> {
    let mut walk_builder = WalkBuilder::new(search_path);
    walk_builder
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true);

    if let Some(glob) = include {
        let mut ob = OverrideBuilder::new(search_path);
        ob.add(glob).map_err(|e| ToolError::InvalidInput {
            name: tool_name.to_string(),
            reason: format!("invalid include glob '{}': {}", glob, e),
        })?;
        let overrides = ob.build().map_err(|e| ToolError::InvalidInput {
            name: tool_name.to_string(),
            reason: format!("cannot build include glob '{}': {}", glob, e),
        })?;
        walk_builder.overrides(overrides);
    }

    let mut searcher = SearcherBuilder::new().build();
    // Min-heap keyed by Reverse<mtime>: peek() yields the oldest entry, enabling O(limit) memory.
    let mut heap: BinaryHeap<(Reverse<SystemTime>, PathBuf)> = BinaryHeap::new();
    let mut total = 0usize;

    for entry in walk_builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(true) {
            continue;
        }
        let path = entry.path();
        let mut sink = MatchSink { matched: false };
        if searcher.search_path(matcher, path, &mut sink).is_err() {
            continue;
        }
        if sink.matched {
            total += 1;
            let mtime = path
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let rel = path.strip_prefix(working_dir).unwrap_or(path).to_path_buf();
            if heap.len() < limit {
                heap.push((Reverse(mtime), rel));
            } else if heap.peek().is_some_and(|(rev_t, _)| mtime > rev_t.0) {
                heap.pop();
                heap.push((Reverse(mtime), rel));
            }
        }
    }

    let truncated = total > limit;
    let mut sorted: Vec<(Reverse<SystemTime>, PathBuf)> = heap.into_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let files: Vec<String> = sorted
        .into_iter()
        .map(|(_, p)| p.to_string_lossy().into_owned())
        .collect();

    Ok(serde_json::json!({
        "files": files,
        "total": total,
        "truncated": truncated,
    }))
}

struct MatchSink {
    matched: bool,
}

impl Sink for MatchSink {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _: &grep_searcher::Searcher,
        _: &SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        self.matched = true;
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tool_in(dir: &TempDir) -> GrepFilesTool {
        GrepFilesTool::with_working_dir(dir.path())
    }

    fn write(dir: &TempDir, name: &str, content: &str) {
        std::fs::write(dir.path().join(name), content).unwrap();
    }

    fn input(
        pattern: &str,
        path: Option<&str>,
        include: Option<&str>,
        limit: Option<usize>,
    ) -> Value {
        serde_json::json!({
            "pattern": pattern,
            "path": path,
            "include": include,
            "limit": limit,
        })
    }

    #[tokio::test]
    async fn returns_matching_files() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "match.rs", "fn main() {}");
        write(&dir, "no_match.rs", "fn helper() {}");

        let result = tool_in(&dir)
            .execute(input("fn main", None, None, None))
            .await
            .unwrap();

        let files = result["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].as_str().unwrap().contains("match.rs"));
        assert_eq!(result["total"], 1);
        assert_eq!(result["truncated"], false);
    }

    #[tokio::test]
    async fn no_matches_returns_empty_ok() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "a.rs", "fn main() {}");

        let result = tool_in(&dir)
            .execute(input("xyz_nonexistent_pattern", None, None, None))
            .await
            .unwrap();

        let files = result["files"].as_array().unwrap();
        assert_eq!(files.len(), 0);
        assert_eq!(result["total"], 0);
        assert_eq!(result["truncated"], false);
    }

    #[tokio::test]
    async fn include_glob_filters_by_extension() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "a.rs", "hello");
        write(&dir, "b.toml", "hello");

        let result = tool_in(&dir)
            .execute(input("hello", None, Some("*.rs"), None))
            .await
            .unwrap();

        let files = result["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].as_str().unwrap().ends_with(".rs"));
    }

    #[tokio::test]
    async fn limit_truncates_and_sets_flag() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..5 {
            write(&dir, &format!("f{i}.txt"), "needle");
        }

        let result = tool_in(&dir)
            .execute(input("needle", None, None, Some(2)))
            .await
            .unwrap();

        assert_eq!(result["files"].as_array().unwrap().len(), 2);
        assert_eq!(result["total"], 5);
        assert_eq!(result["truncated"], true);
    }

    #[tokio::test]
    async fn invalid_regex_returns_invalid_input() {
        let dir = tempfile::tempdir().unwrap();
        let err = tool_in(&dir)
            .execute(input("[unclosed", None, None, None))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn path_outside_working_dir_returns_execution_failed() {
        let dir = tempfile::tempdir().unwrap();
        let err = tool_in(&dir)
            .execute(input("foo", Some("/tmp"), None, None))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed { .. }));
    }

    #[tokio::test]
    async fn paths_returned_relative_to_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir, "nested.rs", "pattern_xyz");

        let result = tool_in(&dir)
            .execute(input("pattern_xyz", None, None, None))
            .await
            .unwrap();

        let files = result["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        let path = files[0].as_str().unwrap();
        assert!(
            !path.starts_with('/'),
            "path should be relative, got: {path}"
        );
        assert!(path.ends_with("nested.rs"));
    }

    #[test]
    fn input_schema_strips_draft_metadata() {
        let schema = GrepFilesTool::new().input_schema();
        let obj = schema.as_object().expect("schema must be an object");
        assert!(!obj.contains_key("$schema"));
        assert!(!obj.contains_key("title"));
        assert!(obj.contains_key("properties"));
    }

    #[test]
    fn input_schema_has_pattern_as_required() {
        let schema = GrepFilesTool::new().input_schema();
        let required = schema["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("pattern")));
    }
}
