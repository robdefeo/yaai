use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use portable_pty::{native_pty_system, CommandBuilder, PtySize};

fn spawn_tui() -> anyhow::Result<PtyHarness> {
    let pty_system = native_pty_system();
    let pair = pty_system.openpty(PtySize {
        rows: 24,
        cols: 100,
        pixel_width: 0,
        pixel_height: 0,
    })?;

    let mut command = CommandBuilder::new(env!("CARGO_BIN_EXE_yaai"));
    command.arg("--model");
    command.arg("openai/gpt-4o");
    command.env("OPENAI_API_KEY", "test-key");

    let child = pair.slave.spawn_command(command)?;
    drop(pair.slave);

    let mut reader = pair.master.try_clone_reader()?;
    let writer = pair.master.take_writer()?;
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let output_clone = Arc::clone(&output);

    let reader_thread = thread::spawn(move || {
        let mut buf = [0_u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => output_clone.lock().unwrap().extend_from_slice(&buf[..n]),
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    });

    Ok(PtyHarness {
        child,
        writer,
        output,
        reader_thread: Some(reader_thread),
    })
}

struct PtyHarness {
    child: Box<dyn portable_pty::Child + Send>,
    writer: Box<dyn Write + Send>,
    output: Arc<Mutex<Vec<u8>>>,
    reader_thread: Option<thread::JoinHandle<()>>,
}

impl PtyHarness {
    fn send_str(&mut self, input: &str) -> anyhow::Result<()> {
        self.writer.write_all(input.as_bytes())?;
        self.writer.flush()?;
        Ok(())
    }

    fn send_ctrl_c(&mut self) -> anyhow::Result<()> {
        self.writer.write_all(&[0x03])?;
        self.writer.flush()?;
        Ok(())
    }

    fn wait_for_screen_text(&self, needle: &str, timeout: Duration) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if self.snapshot().contains(needle) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(25));
        }

        anyhow::bail!("timed out waiting for screen text: {needle}");
    }

    fn snapshot(&self) -> String {
        let bytes = self.output.lock().unwrap().clone();
        let mut parser = vt100::Parser::new(24, 100, 0);
        parser.process(&bytes);
        parser.screen().contents().to_string()
    }

    fn wait_for_exit(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if self.child.try_wait()?.is_some() {
                if let Some(reader_thread) = self.reader_thread.take() {
                    reader_thread.join().unwrap();
                }
                return Ok(());
            }
            thread::sleep(Duration::from_millis(25));
        }

        self.child.kill()?;
        anyhow::bail!("timed out waiting for process exit");
    }
}

impl Drop for PtyHarness {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader_thread) = self.reader_thread.take() {
            let _ = reader_thread.join();
        }
    }
}

#[test]
fn tui_renders_initial_screen_in_a_pty() {
    let mut harness = spawn_tui().unwrap();

    harness
        .wait_for_screen_text("Transcript", Duration::from_secs(5))
        .unwrap();
    harness
        .wait_for_screen_text("Prompt", Duration::from_secs(5))
        .unwrap();

    harness.send_ctrl_c().unwrap();
    harness.wait_for_exit(Duration::from_secs(5)).unwrap();
}

#[test]
fn tui_accepts_typed_input_before_submit() {
    let mut harness = spawn_tui().unwrap();

    harness
        .wait_for_screen_text("Prompt", Duration::from_secs(5))
        .unwrap();
    harness.send_str("hello from pty").unwrap();
    harness
        .wait_for_screen_text("hello from pty", Duration::from_secs(5))
        .unwrap();

    harness.send_ctrl_c().unwrap();
    harness.wait_for_exit(Duration::from_secs(5)).unwrap();
}

#[test]
fn tui_exits_with_zero_status_on_ctrl_c() {
    let mut harness = spawn_tui().unwrap();
    harness
        .wait_for_screen_text("Transcript", Duration::from_secs(5))
        .unwrap();
    harness.send_ctrl_c().unwrap();
    harness.wait_for_exit(Duration::from_secs(5)).unwrap();

    let status = harness.child.wait().unwrap();
    assert_eq!(
        status.exit_code(),
        0,
        "process should exit cleanly after Ctrl-C"
    );
}

#[test]
fn tui_esc_restores_ready_status() {
    let mut harness = spawn_tui().unwrap();
    harness
        .wait_for_screen_text("Ready.", Duration::from_secs(5))
        .unwrap();

    // Esc should keep (or restore) the Ready status
    harness.send_str("\x1b").unwrap();
    harness
        .wait_for_screen_text("Ready.", Duration::from_secs(5))
        .unwrap();

    harness.send_ctrl_c().unwrap();
    harness.wait_for_exit(Duration::from_secs(5)).unwrap();
}
