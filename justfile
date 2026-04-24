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

# Generate HTML + lcov coverage reports, then check thresholds (lines >= 80%, functions >= 20%);
# on failure show per-file breakdown sorted by worst coverage
coverage:
  mkdir -p coverage
  cargo llvm-cov --workspace --no-report
  cargo llvm-cov report --lcov --output-path coverage/lcov --ignore-filename-regex 'apps/cli/src/main\.rs'
  cargo llvm-cov report --html --output-dir coverage --ignore-filename-regex 'apps/cli/src/main\.rs'
  printf "\n  %-50s  %8s  %-10s  %9s  %-10s\n" "File" "Lines" "(hit/tot)" "Functions" "(hit/tot)"
  printf "  %-50s  %8s  %-10s  %9s  %-10s\n" "--------------------------------------------------" "--------" "----------" "---------" "----------"
  awk -F: '\
    /^SF:/  { file=$2; lh=0; lf=0; fh=0; ff=0 }\
    /^LH:/  { lh=$2 } /^LF:/ { lf=$2 }\
    /^FNH:/ { fh=$2 } /^FNF:/ { ff=$2 }\
    /^end_of_record/ {\
      l=(lf>0)?(lh/lf*100):0; f=(ff>0)?(fh/ff*100):0;\
      if(l<80||f<20) printf "%06.2f %-50s  %7.1f%%  (%3d/%-3d)  %8.1f%%  (%2d/%-2d)\n",l,file,l,lh,lf,f,fh,ff\
    }' coverage/lcov | sort -k1 -n | sed 's/^[0-9.]* /  /'
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
    }' coverage/lcov

# Clean build artifacts
clean:
  cargo clean
  rm -rf coverage/ traces/
