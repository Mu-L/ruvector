//! Download small GGUF models for testing
//!
//! This utility downloads small, quantized models suitable for testing RuvLLM.
//!
//! ## Usage
//!
//! ```bash
//! # Download TinyLlama (recommended for quick tests)
//! cargo run -p ruvllm --example download_test_model -- --model tinyllama
//!
//! # Download Qwen2-0.5B (smallest, fastest)
//! cargo run -p ruvllm --example download_test_model -- --model qwen-0.5b
//!
//! # Download to custom directory
//! cargo run -p ruvllm --example download_test_model -- --model tinyllama --output ./my_models
//!
//! # List available models
//! cargo run -p ruvllm --example download_test_model -- --list
//! ```
//!
//! ## Available Models
//!
//! | Model | Size | Download Time | Use Case |
//! |-------|------|---------------|----------|
//! | tinyllama | ~600MB | ~2-5 min | Fast iteration, general testing |
//! | qwen-0.5b | ~400MB | ~1-3 min | Smallest, fastest tests |
//! | phi-3-mini | ~2.2GB | ~10-20 min | Higher quality outputs |
//! | gemma-2b | ~1.5GB | ~5-10 min | Google's efficient model |
//!
//! ## Environment Variables
//!
//! - `HF_TOKEN`: HuggingFace token for gated models (optional for most models)
//! - `RUVLLM_MODELS_DIR`: Default output directory for models

use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Model definitions with HuggingFace URLs
const MODELS: &[ModelDef] = &[
    ModelDef {
        name: "tinyllama",
        display_name: "TinyLlama 1.1B Chat Q4_K_M",
        url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        filename: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        size_mb: 669,
        architecture: "llama",
        description: "Fast, small model ideal for testing. Good general performance.",
    },
    ModelDef {
        name: "qwen-0.5b",
        display_name: "Qwen2 0.5B Instruct Q4_K_M",
        url: "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf",
        filename: "qwen2-0_5b-instruct-q4_k_m.gguf",
        size_mb: 400,
        architecture: "qwen2",
        description: "Smallest recommended model. Excellent for quick iteration.",
    },
    ModelDef {
        name: "phi-3-mini",
        display_name: "Phi-3 Mini 4K Instruct Q4_K_M",
        url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        filename: "Phi-3-mini-4k-instruct-q4.gguf",
        size_mb: 2200,
        architecture: "phi3",
        description: "Microsoft's efficient model. Higher quality outputs.",
    },
    ModelDef {
        name: "gemma-2b",
        display_name: "Gemma 2B Instruct Q4_K_M",
        url: "https://huggingface.co/google/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_M.gguf",
        filename: "gemma-2b-it.Q4_K_M.gguf",
        size_mb: 1500,
        architecture: "gemma",
        description: "Google's efficient model with good instruction following.",
    },
    ModelDef {
        name: "stablelm-2-1.6b",
        display_name: "StableLM 2 1.6B Chat Q4_K_M",
        url: "https://huggingface.co/TheBloke/stablelm-2-1_6b-chat-GGUF/resolve/main/stablelm-2-1_6b-chat.Q4_K_M.gguf",
        filename: "stablelm-2-1_6b-chat.Q4_K_M.gguf",
        size_mb: 1000,
        architecture: "stablelm",
        description: "Stability AI's efficient chat model.",
    },
];

struct ModelDef {
    name: &'static str,
    display_name: &'static str,
    url: &'static str,
    filename: &'static str,
    size_mb: usize,
    architecture: &'static str,
    description: &'static str,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        print_help();
        return;
    }

    if args.contains(&"--list".to_string()) || args.contains(&"-l".to_string()) {
        list_models();
        return;
    }

    // Parse arguments
    let mut model_name: Option<&str> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut force = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_name = Some(args[i].as_str());
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_dir = Some(PathBuf::from(&args[i]));
                }
            }
            "--force" | "-f" => {
                force = true;
            }
            arg if !arg.starts_with('-') && model_name.is_none() => {
                model_name = Some(arg);
            }
            _ => {}
        }
        i += 1;
    }

    let model_name = match model_name {
        Some(name) => name,
        None => {
            eprintln!("Error: No model specified.");
            eprintln!("Use --list to see available models.");
            std::process::exit(1);
        }
    };

    // Find the model definition
    let model = match MODELS.iter().find(|m| m.name == model_name) {
        Some(m) => m,
        None => {
            eprintln!("Error: Unknown model '{}'", model_name);
            eprintln!("Available models:");
            for m in MODELS {
                eprintln!("  - {}", m.name);
            }
            std::process::exit(1);
        }
    };

    // Determine output directory
    let output_dir = output_dir
        .or_else(|| env::var("RUVLLM_MODELS_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("./test_models"));

    // Create output directory
    if let Err(e) = fs::create_dir_all(&output_dir) {
        eprintln!("Error creating output directory: {}", e);
        std::process::exit(1);
    }

    let output_path = output_dir.join(model.filename);

    // Check if file already exists
    if output_path.exists() && !force {
        println!("Model already exists: {}", output_path.display());
        println!("Use --force to re-download.");

        // Verify file size
        if let Ok(metadata) = fs::metadata(&output_path) {
            let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
            let expected_mb = model.size_mb as f64;
            if (size_mb - expected_mb).abs() / expected_mb > 0.1 {
                println!("Warning: File size ({:.1} MB) differs from expected ({} MB)", size_mb, model.size_mb);
                println!("Consider re-downloading with --force");
            } else {
                println!("File size verified: {:.1} MB", size_mb);
            }
        }
        return;
    }

    // Print download info
    println!("Downloading: {}", model.display_name);
    println!("Architecture: {}", model.architecture);
    println!("Size: ~{} MB", model.size_mb);
    println!("Destination: {}", output_path.display());
    println!();

    // Estimate download time
    let estimated_time = estimate_download_time(model.size_mb);
    println!("Estimated download time: {}", format_duration(estimated_time));
    println!();

    // Download the model
    match download_model(model.url, &output_path, model.size_mb) {
        Ok(()) => {
            println!("\nDownload complete!");
            println!("Model saved to: {}", output_path.display());
            println!();
            println!("To run tests with this model:");
            println!("  TEST_MODEL_PATH={} cargo test -p ruvllm --test real_model_test -- --ignored",
                     output_path.display());
        }
        Err(e) => {
            eprintln!("\nDownload failed: {}", e);
            // Clean up partial download
            let _ = fs::remove_file(&output_path);
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!("RuvLLM Test Model Downloader");
    println!();
    println!("USAGE:");
    println!("    cargo run -p ruvllm --example download_test_model -- [OPTIONS] <MODEL>");
    println!();
    println!("ARGUMENTS:");
    println!("    <MODEL>    Model to download (use --list to see options)");
    println!();
    println!("OPTIONS:");
    println!("    -m, --model <MODEL>     Model to download");
    println!("    -o, --output <DIR>      Output directory (default: ./test_models)");
    println!("    -f, --force             Force re-download even if file exists");
    println!("    -l, --list              List available models");
    println!("    -h, --help              Print help information");
    println!();
    println!("ENVIRONMENT VARIABLES:");
    println!("    HF_TOKEN              HuggingFace token for gated models");
    println!("    RUVLLM_MODELS_DIR     Default output directory");
    println!();
    println!("EXAMPLES:");
    println!("    # Download TinyLlama (recommended for quick tests)");
    println!("    cargo run -p ruvllm --example download_test_model -- tinyllama");
    println!();
    println!("    # Download to custom directory");
    println!("    cargo run -p ruvllm --example download_test_model -- -m qwen-0.5b -o ./models");
}

fn list_models() {
    println!("Available models for testing:\n");
    println!("{:<15} {:>8}  {:<40}", "NAME", "SIZE", "DESCRIPTION");
    println!("{}", "-".repeat(70));

    for model in MODELS {
        println!(
            "{:<15} {:>6}MB  {}",
            model.name,
            model.size_mb,
            model.description
        );
    }

    println!();
    println!("Recommendations:");
    println!("  - For quick tests: tinyllama or qwen-0.5b");
    println!("  - For quality testing: phi-3-mini");
    println!("  - For architecture variety: download multiple models");
}

fn estimate_download_time(size_mb: usize) -> Duration {
    // Assume ~10 MB/s average download speed
    let speed_mbps = 10.0;
    let seconds = size_mb as f64 / speed_mbps;
    Duration::from_secs_f64(seconds)
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{} seconds", secs)
    } else if secs < 3600 {
        format!("{} min {} sec", secs / 60, secs % 60)
    } else {
        format!("{} hr {} min", secs / 3600, (secs % 3600) / 60)
    }
}

fn download_model(url: &str, output_path: &Path, expected_size_mb: usize) -> io::Result<()> {
    // Use curl or wget if available, otherwise fall back to pure Rust
    if which_cmd("curl") {
        download_with_curl(url, output_path, expected_size_mb)
    } else if which_cmd("wget") {
        download_with_wget(url, output_path)
    } else {
        download_with_rust(url, output_path, expected_size_mb)
    }
}

fn which_cmd(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn download_with_curl(url: &str, output_path: &Path, _expected_size_mb: usize) -> io::Result<()> {
    println!("Downloading with curl...");

    let status = std::process::Command::new("curl")
        .args([
            "-L",                    // Follow redirects
            "-#",                    // Progress bar
            "--fail",                // Fail on HTTP errors
            "-o", output_path.to_str().unwrap(),
            url,
        ])
        .status()?;

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("curl exited with status: {}", status),
        ))
    }
}

fn download_with_wget(url: &str, output_path: &Path) -> io::Result<()> {
    println!("Downloading with wget...");

    let status = std::process::Command::new("wget")
        .args([
            "-q",                    // Quiet
            "--show-progress",       // But show progress
            "-O", output_path.to_str().unwrap(),
            url,
        ])
        .status()?;

    if status.success() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            format!("wget exited with status: {}", status),
        ))
    }
}

fn download_with_rust(url: &str, output_path: &Path, _expected_size_mb: usize) -> io::Result<()> {
    println!("Downloading with built-in HTTP client...");
    println!("Note: For faster downloads, install curl or wget.");

    // Simple HTTP download using std library
    // This is a basic implementation - production code should use reqwest or similar

    let url_parts: Vec<&str> = url.split('/').collect();
    let _host = url_parts.get(2).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Invalid URL")
    })?;

    let _path = format!("/{}", url_parts[3..].join("/"));

    // For HTTPS, we need to use a TLS library
    // This simple example shows the structure but won't work for HTTPS
    println!("Warning: Built-in downloader doesn't support HTTPS.");
    println!("Please install curl: brew install curl (macOS) or apt install curl (Linux)");

    // Create a placeholder file to show where the model should go
    let mut file = BufWriter::new(File::create(output_path)?);
    writeln!(file, "# Placeholder - download failed")?;
    writeln!(file, "# Download manually from: {}", url)?;
    writeln!(file, "# Or install curl and re-run this command")?;

    Err(io::Error::new(
        io::ErrorKind::Other,
        "HTTPS download requires curl or wget. Please install curl.",
    ))
}

/// Format bytes with appropriate unit
#[allow(dead_code)]
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.46 KB");
        assert_eq!(format_bytes(1_500_000), "1.43 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.40 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30 seconds");
        assert_eq!(format_duration(Duration::from_secs(90)), "1 min 30 sec");
        assert_eq!(format_duration(Duration::from_secs(3700)), "1 hr 1 min");
    }

    #[test]
    fn test_model_definitions() {
        // Verify all models have valid data
        for model in MODELS {
            assert!(!model.name.is_empty());
            assert!(!model.url.is_empty());
            assert!(model.url.starts_with("https://"));
            assert!(model.size_mb > 0);
            assert!(model.filename.ends_with(".gguf"));
        }
    }
}
