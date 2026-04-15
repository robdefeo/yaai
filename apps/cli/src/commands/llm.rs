use anyhow::{anyhow, bail, Context, Result};
use yaai_llm::{AnthropicClient, LlmClient, OpenAiClient};

#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    OpenAi,
    Anthropic,
}

pub fn parse_provider_model(s: &str) -> Result<(Provider, String)> {
    let (provider_str, model) = s
        .split_once('/')
        .ok_or_else(|| anyhow!("--model must be in format provider/model (e.g. openai/gpt-4o)"))?;
    let provider_str = provider_str.trim();
    let model = model.trim();

    if provider_str.is_empty() || model.is_empty() {
        bail!("--model must include non-empty provider and model segments");
    }
    let provider = match provider_str {
        "openai" => Provider::OpenAi,
        "anthropic" => Provider::Anthropic,
        other => bail!("unknown provider '{other}', expected 'openai' or 'anthropic'"),
    };
    Ok((provider, model.to_string()))
}

pub fn validate_api_key(key: &str, var_name: &str) -> Result<()> {
    if key.trim().is_empty() {
        bail!("{var_name} must not be empty");
    }
    Ok(())
}

fn normalize_api_key(key: String, var_name: &str) -> Result<String> {
    let key = key.trim().to_string();
    validate_api_key(&key, var_name)?;
    Ok(key)
}

pub fn build_llm_client(provider: &Provider, model: &str) -> Result<Box<dyn LlmClient>> {
    match provider {
        Provider::OpenAi => {
            let api_key =
                std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY env var not set")?;
            let api_key = normalize_api_key(api_key, "OPENAI_API_KEY")?;
            Ok(Box::new(OpenAiClient::new(api_key, model)))
        }
        Provider::Anthropic => {
            let api_key =
                std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY env var not set")?;
            let api_key = normalize_api_key(api_key, "ANTHROPIC_API_KEY")?;
            Ok(Box::new(AnthropicClient::new(api_key, model)))
        }
    }
}

// grcov-excl-start: exclude inline unit tests from production coverage
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_env_var<T>(name: &str, value: Option<&str>, test: impl FnOnce() -> T) -> T {
        let _guard = env_lock().lock().unwrap();
        let previous = std::env::var_os(name);
        unsafe {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        let result = test();
        unsafe {
            match previous {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        result
    }

    #[test]
    fn parse_openai_model() {
        let (provider, model) = parse_provider_model("openai/gpt-4o").unwrap();
        assert_eq!(provider, Provider::OpenAi);
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn parse_provider_model_trims_segments() {
        let (provider, model) = parse_provider_model("  openai  /  gpt-4o  ").unwrap();
        assert_eq!(provider, Provider::OpenAi);
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn parse_anthropic_model() {
        let (provider, model) =
            parse_provider_model("anthropic/claude-3-5-sonnet-20241022").unwrap();
        assert_eq!(provider, Provider::Anthropic);
        assert_eq!(model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn parse_missing_slash_fails() {
        let err = parse_provider_model("gpt-4o").unwrap_err();
        assert!(err.to_string().contains("provider/model"));
    }

    #[test]
    fn parse_unknown_provider_fails() {
        let err = parse_provider_model("bedrock/titan").unwrap_err();
        assert!(err.to_string().contains("unknown provider"));
    }

    #[test]
    fn parse_empty_model_fails() {
        let err = parse_provider_model("openai/").unwrap_err();
        assert!(err.to_string().contains("non-empty provider and model"));
    }

    #[test]
    fn parse_whitespace_model_fails() {
        let err = parse_provider_model("openai/   ").unwrap_err();
        assert!(err.to_string().contains("non-empty provider and model"));
    }

    #[test]
    fn validate_api_key_rejects_empty() {
        let err = validate_api_key("", "MY_KEY").unwrap_err();
        assert!(err.to_string().contains("MY_KEY must not be empty"));
    }

    #[test]
    fn validate_api_key_rejects_whitespace() {
        let err = validate_api_key("   ", "MY_KEY").unwrap_err();
        assert!(err.to_string().contains("MY_KEY must not be empty"));
    }

    #[test]
    fn validate_api_key_accepts_valid() {
        assert!(validate_api_key("sk-abc123", "MY_KEY").is_ok());
    }

    #[test]
    fn normalize_api_key_trims_value() {
        let key = normalize_api_key("  sk-abc123\n".to_string(), "MY_KEY").unwrap();
        assert_eq!(key, "sk-abc123");
    }

    #[test]
    fn normalize_api_key_rejects_whitespace_only() {
        let err = normalize_api_key("   \n\t".to_string(), "MY_KEY").unwrap_err();
        assert!(err.to_string().contains("MY_KEY must not be empty"));
    }

    #[test]
    fn build_openai_client_requires_api_key() {
        let err = with_env_var("OPENAI_API_KEY", None, || {
            match build_llm_client(&Provider::OpenAi, "gpt-4o") {
                Ok(_) => panic!("expected OPENAI_API_KEY lookup to fail"),
                Err(err) => err,
            }
        });
        assert!(err.to_string().contains("OPENAI_API_KEY env var not set"));
    }

    #[test]
    fn build_openai_client_rejects_whitespace_api_key() {
        let err = with_env_var("OPENAI_API_KEY", Some("   "), || {
            match build_llm_client(&Provider::OpenAi, "gpt-4o") {
                Ok(_) => panic!("expected whitespace OPENAI_API_KEY to fail"),
                Err(err) => err,
            }
        });
        assert!(err.to_string().contains("OPENAI_API_KEY must not be empty"));
    }

    #[test]
    fn build_openai_client_accepts_valid_api_key() {
        with_env_var("OPENAI_API_KEY", Some("sk-openai"), || {
            assert!(build_llm_client(&Provider::OpenAi, "gpt-4o").is_ok());
        });
    }

    #[test]
    fn build_anthropic_client_requires_api_key() {
        let err = with_env_var("ANTHROPIC_API_KEY", None, || {
            match build_llm_client(&Provider::Anthropic, "claude-3-5-sonnet-20241022") {
                Ok(_) => panic!("expected ANTHROPIC_API_KEY lookup to fail"),
                Err(err) => err,
            }
        });
        assert!(err
            .to_string()
            .contains("ANTHROPIC_API_KEY env var not set"));
    }

    #[test]
    fn build_anthropic_client_rejects_whitespace_api_key() {
        let err = with_env_var("ANTHROPIC_API_KEY", Some("   "), || match build_llm_client(
            &Provider::Anthropic,
            "claude-3-5-sonnet-20241022",
        ) {
            Ok(_) => panic!("expected whitespace ANTHROPIC_API_KEY to fail"),
            Err(err) => err,
        });
        assert!(err
            .to_string()
            .contains("ANTHROPIC_API_KEY must not be empty"));
    }

    #[test]
    fn build_anthropic_client_accepts_valid_api_key() {
        with_env_var("ANTHROPIC_API_KEY", Some("sk-anthropic"), || {
            assert!(build_llm_client(&Provider::Anthropic, "claude-3-5-sonnet-20241022").is_ok());
        });
    }
}
// grcov-excl-stop
