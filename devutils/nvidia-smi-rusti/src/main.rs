use clap::Parser;
use regex::Regex;
use std::process::Command;
use std::process;

/// Utility for monitoring Burn (Rust) processes on NVIDIA GPUs using nvidia-smi
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Filter by process name (e.g. 'burn', 'rust', or your binary name)
    #[arg(short, long, default_value = "")] 
    name: String,
    /// Filter by PID (optional)
    #[arg(short, long)]
    pid: Option<u32>,
    /// Show all GPU processes (ignore filters)
    #[arg(long)]
    all: bool,
}

fn main() {
    let args = Args::parse();
    let output = Command::new("nvidia-smi")
        .arg("--query-compute-apps=pid,process_name,used_memory,gpu_uuid")
        .arg("--format=csv,noheader,nounits")
        .output()
        .expect("Failed to run nvidia-smi");

    if !output.status.success() {
        eprintln!("nvidia-smi failed: {}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines = stdout.lines();
    let name_re = if args.name.is_empty() {
        None
    } else {
        Some(Regex::new(&args.name).unwrap())
    };

    println!("{:<8} {:<20} {:<10} GPU UUID", "PID", "Process", "Mem(MB)");
    for line in lines {
        let fields: Vec<_> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 4 { continue; }
        let pid: u32 = fields[0].parse().unwrap_or(0);
        let pname = fields[1];
        let mem = fields[2];
        let gpu = fields[3];
        if args.all
            || (args.pid.is_none_or(|p| p == pid))
            || (name_re.as_ref().is_some_and(|re| re.is_match(pname)))
        {
            println!("{pid:<8} {pname:<20} {mem:<10} {gpu}");
        }
    }
}
