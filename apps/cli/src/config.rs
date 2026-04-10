//! User-level configuration loaded from the OS config directory (e.g.
//! `~/.config/yaai/config.json` on Linux, `~/Library/Application Support/yaai/config.json`
//! on macOS). Use [`config_path`] to get the exact path at runtime.
//!
//! All fields are optional — the file need not exist, and any field can be
//! omitted. CLI arguments always take precedence over file values.

use anyhow::{Context, Result};
use config::{Config, File, FileFormat};
use serde::Deserialize;

/// Fields that can be set in the yaai config file (see [`config_path`]).
/// Every field is `Option<T>` so partial configs are valid.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct YaaiConfig {
    /// Default model in `provider/model` format (e.g. `"openai/gpt-4o"`).
    pub model: Option<String>,

    /// Directory where trace NDJSON files are written.
    pub traces_dir: Option<String>,

    /// Emit logs as structured JSON instead of pretty-printed text.
    pub json_logs: Option<bool>,
}

/// Returns the path where the config file is expected, if the OS config dir is available.
pub fn config_path() -> Option<std::path::PathBuf> {
    dirs::config_dir().map(|d| d.join("yaai").join("config.json"))
}

/// Returns [`config_path`] as a display string with the home directory replaced by `~`.
pub fn config_path_display() -> String {
    let Some(path) = config_path() else {
        return "the yaai config file".to_string();
    };
    if let Some(home) = dirs::home_dir() {
        if let Ok(rel) = path.strip_prefix(&home) {
            return format!("~/{}", rel.display());
        }
    }
    path.display().to_string()
}

/// Load the config file if it exists; return `YaaiConfig::default()` otherwise.
pub fn load() -> Result<YaaiConfig> {
    let Some(config_dir) = dirs::config_dir() else {
        return Ok(YaaiConfig::default());
    };

    let path = config_dir.join("yaai").join("config.json");

    if !path.exists() {
        return Ok(YaaiConfig::default());
    }

    let path_str = path.to_string_lossy();

    let cfg = Config::builder()
        .add_source(File::new(&path_str, FileFormat::Json5))
        .build()
        .with_context(|| format!("failed to parse config file: {path_str}"))?;

    cfg.try_deserialize::<YaaiConfig>()
        .with_context(|| format!("invalid config file: {path_str}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn build_config(json: &str) -> Result<YaaiConfig> {
        let dir = tempdir().unwrap();
        let file = dir.path().join("config.json");
        fs::write(&file, json).unwrap();

        let path_str = file.to_string_lossy().to_string();
        let cfg = Config::builder()
            .add_source(File::new(&path_str, FileFormat::Json5))
            .build()?;
        Ok(cfg.try_deserialize::<YaaiConfig>()?)
    }

    #[test]
    fn empty_object_yields_all_none() {
        let c = build_config("{}").unwrap();
        assert!(c.model.is_none());
        assert!(c.traces_dir.is_none());
        assert!(c.json_logs.is_none());
    }

    #[test]
    fn partial_config_deserializes() {
        let c = build_config(r#"{"model": "openai/gpt-4o"}"#).unwrap();
        assert_eq!(c.model.as_deref(), Some("openai/gpt-4o"));
        assert!(c.traces_dir.is_none());
    }

    #[test]
    fn full_config_deserializes() {
        let c = build_config(
            r#"{"model":"anthropic/claude-3-5-sonnet-20241022","traces_dir":"/tmp/traces","json_logs":true}"#,
        )
        .unwrap();
        assert_eq!(
            c.model.as_deref(),
            Some("anthropic/claude-3-5-sonnet-20241022")
        );
        assert_eq!(c.traces_dir.as_deref(), Some("/tmp/traces"));
        assert_eq!(c.json_logs, Some(true));
    }

    #[test]
    fn unknown_field_is_rejected() {
        let err = build_config(r#"{"unknown_key": "value"}"#).unwrap_err();
        assert!(err.to_string().contains("unknown") || err.to_string().contains("unknown_key"));
    }

    #[test]
    fn json5_comments_and_trailing_commas_are_valid() {
        let c = build_config(
            r#"{
                // default model
                "model": "openai/gpt-4o",
                /* output directory */
                "traces_dir": "/tmp/traces",
                "json_logs": false, // trailing comma
            }"#,
        )
        .unwrap();
        assert_eq!(c.model.as_deref(), Some("openai/gpt-4o"));
        assert_eq!(c.traces_dir.as_deref(), Some("/tmp/traces"));
        assert_eq!(c.json_logs, Some(false));
    }

    #[test]
    fn load_returns_default_when_config_file_absent() {
        let dir = tempdir().unwrap();
        // Point HOME (and XDG_CONFIG_HOME) at an empty temp dir so no config file exists.
        unsafe {
            std::env::set_var("HOME", dir.path());
            std::env::set_var("XDG_CONFIG_HOME", dir.path().join("config"));
        }
        let cfg = load().unwrap();
        assert!(cfg.model.is_none());
        assert!(cfg.traces_dir.is_none());
        assert!(cfg.json_logs.is_none());
    }
}
