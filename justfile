set shell := ["bash", "-ceu"]

# Install all tools via mise
install:
  mise trust --yes
  mise install
  bun install

# Build all Rust crates
build:
  cargo build --workspace

# Run all tests (Rust + Bun)
test:
  cargo test --workspace
  bun run test

# Lint: rustfmt check + clippy -D warnings + biome
lint:
  cargo fmt --check --all -- --config-path .config
  cargo clippy --workspace --all-targets -- -D warnings
  bun run lint

# Format: rustfmt + biome
fmt:
  cargo fmt --all -- --config-path .config
  bun run fmt

run:
  cargo run -p yaai

# Start the agent CLI in dev/watch mode
dev:
  mise exec cargo:cargo-watch -- cargo watch -x 'run -p yaai'

# [private] Collect profraw data and run grcov with the given format and output path
_grcov format output:
  mkdir -p coverage coverage/rust-profraw
  find coverage/rust-profraw -name '*.profraw' -delete
  CARGO_BUILD_JOBS=1 \
  CARGO_INCREMENTAL=0 \
  RUSTFLAGS="-Cinstrument-coverage" \
  LLVM_PROFILE_FILE="$(pwd)/coverage/rust-profraw/yaai-%p-%m.profraw" \
  cargo test --workspace
  mise exec -- grcov coverage/rust-profraw \
    --binary-path ./target/debug/deps \
    --llvm-path "$(dirname "$(mise exec -- rustc --print target-libdir)")/bin" \
    -s . \
    --branch \
    --ignore-not-existing \
    --ignore "${HOME}/.cargo/*" \
    --ignore "${HOME}/.rustup/*" \
    --ignore "*/tests/*" \
    --excl-line "#\[derive\(|grcov-excl-line" \
    --excl-start "grcov-excl-start" \
    --excl-stop "grcov-excl-stop" \
    -t {{format}} \
    -o {{output}}

# Generate HTML coverage report
coverage-html: (_grcov "html" "coverage/html")

# Generate lcov report and check thresholds (lines >= 80%, functions >= 20%); on failure show per-file breakdown sorted by worst coverage
coverage-check: (_grcov "lcov" "coverage/lcov.info")
  printf "\n  %-50s  %8s  %-10s  %9s  %-10s\n" "File" "Lines" "(hit/tot)" "Functions" "(hit/tot)"
  printf "  %-50s  %8s  %-10s  %9s  %-10s\n" "--------------------------------------------------" "--------" "----------" "---------" "----------"
  awk -F: '\
    /^SF:/  { file=$2; lh=0; lf=0; fh=0; ff=0 }\
    /^LH:/  { lh=$2 } /^LF:/ { lf=$2 }\
    /^FNH:/ { fh=$2 } /^FNF:/ { ff=$2 }\
    /^end_of_record/ {\
      l=(lf>0)?(lh/lf*100):0; f=(ff>0)?(fh/ff*100):0;\
      if(l<80||f<20) printf "%06.2f %-50s  %7.1f%%  (%3d/%-3d)  %8.1f%%  (%2d/%-2d)\n",l,file,l,lh,lf,f,fh,ff\
    }' coverage/lcov.info | sort -k1 -n | sed 's/^[0-9.]* /  /'
  printf "\n"
  awk -F: '\
    /^LH:/{lh+=$2} /^LF:/{lf+=$2}\
    /^FNH:/{fh+=$2} /^FNF:/{ff+=$2}\
    END{\
      l=(lf>0)?(lh/lf*100):0; f=(ff>0)?(fh/ff*100):0;\
      printf "  Total: %.1f%% lines  %.1f%% functions\n\n",l,f;\
      fail=0;\
      if(l<80){printf "  FAIL: lines %.1f%% is below 80%% threshold\n",l; fail=1}\
      if(f<20){printf "  FAIL: functions %.1f%% is below 20%% threshold\n",f; fail=1}\
      exit fail\
    }' coverage/lcov.info

# Clean build artifacts
clean:
  cargo clean
  rm -rf coverage/ traces/
