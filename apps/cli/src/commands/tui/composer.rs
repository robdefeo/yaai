use ratatui::widgets::{Block, Borders};
use tui_textarea::TextArea;

pub(crate) fn new_composer() -> TextArea<'static> {
    let mut composer = TextArea::default();
    composer.set_block(
        Block::default()
            .borders(Borders::ALL)
            .title("Prompt")
            .title_bottom("Enter submit | Shift+Enter newline"),
    );
    composer
}

pub(crate) fn composer_height(composer: &TextArea<'_>, terminal_width: u16) -> u16 {
    let wrap_width = (terminal_width as usize).max(24);
    let wrapped_hint = textwrap::wrap("Enter submit | Shift+Enter newline", wrap_width);
    let body_lines = composer.lines().len().max(1);
    let total = body_lines + wrapped_hint.len() + 2;
    total.min(8) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_height_is_bounded() {
        let mut composer = new_composer();
        composer.insert_str("one\ntwo\nthree\nfour\nfive\nsix\nseven");

        assert_eq!(composer_height(&composer, 80), 8);
    }

    #[test]
    fn new_composer_has_one_empty_line() {
        let composer = new_composer();
        assert_eq!(composer.lines().len(), 1);
        assert_eq!(composer.lines()[0], "");
    }
}
