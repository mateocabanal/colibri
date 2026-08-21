use std::{
    io::{self, IsTerminal, Write},
    time::Instant,
};

use colic::{
    cli::{self, Command, USAGE},
    pipeline::{self, ProgressSink, Stage},
    source::DiscoveryProgress,
    verify::VerificationProgress,
};

const PROGRESS_BAR_WIDTH: usize = 24;

struct ConsoleProgress {
    discovery_started: Option<Instant>,
    emission_started: Option<Instant>,
    verification_started: Option<Instant>,
    interactive: bool,
    active_line_width: usize,
}

impl ConsoleProgress {
    fn new() -> Self {
        Self {
            discovery_started: None,
            emission_started: None,
            verification_started: None,
            interactive: io::stderr().is_terminal(),
            active_line_width: 0,
        }
    }

    fn finish_active_line(&mut self) {
        if self.interactive && self.active_line_width != 0 {
            eprintln!();
            self.active_line_width = 0;
        }
    }

    fn progress_line(&mut self, line: String, complete: bool) {
        if !self.interactive {
            eprintln!("{line}");
            return;
        }

        let padding = self.active_line_width.saturating_sub(line.len());
        eprint!("\r{line}{:padding$}", "");
        let _ = io::stderr().flush();
        self.active_line_width = line.len();
        if complete {
            eprintln!();
            self.active_line_width = 0;
        }
    }

    fn verification(&mut self, update: VerificationProgress) {
        let started = self.verification_started.get_or_insert_with(Instant::now);
        let elapsed = started.elapsed().as_secs_f64();
        let complete = update.completed_records >= update.total_records;
        let percent = progress_percent(update.completed_records, update.total_records);
        let bar = progress_bar(update.completed_records, update.total_records);
        let eta = estimate_eta(update.completed_records, update.total_records, elapsed);
        let throughput = byte_rate(update.verified_bytes, elapsed);
        self.progress_line(
            format!(
                "colic: verify   [{bar}] {percent:5.1}%  {}/{} records  {} checked  {throughput}  ETA {eta}  shard {}/{}",
                update.completed_records,
                update.total_records,
                human_bytes(update.verified_bytes),
                update.current_shard + 1,
                update.total_shards,
            ),
            complete,
        );
    }
}

impl ProgressSink for ConsoleProgress {
    fn stage(&mut self, stage: Stage) {
        self.finish_active_line();
        if stage == Stage::Emission {
            self.emission_started = Some(Instant::now());
        }
        if stage == Stage::Verification {
            self.verification_started = Some(Instant::now());
        }
        eprintln!("colic: {}...", stage.as_str());
    }

    fn emission(&mut self, completed: u64, total: u64, bytes: u64, total_bytes: u64) {
        let elapsed = self
            .emission_started
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let progress_done = if total_bytes != 0 { bytes } else { completed };
        let progress_total = if total_bytes != 0 { total_bytes } else { total };
        let complete = completed >= total && bytes >= total_bytes;
        let percent = progress_percent(progress_done, progress_total);
        let bar = progress_bar(progress_done, progress_total);
        let eta = estimate_eta(progress_done, progress_total, elapsed);
        let throughput = byte_rate(bytes, elapsed);
        self.progress_line(
            format!(
                "colic: emission [{bar}] {percent:5.1}%  {}/{}  {completed}/{total} records  {throughput}  ETA {eta}",
                human_bytes(bytes),
                human_bytes(total_bytes),
            ),
            complete,
        );
    }

    fn source_file(&mut self, update: &DiscoveryProgress) {
        let started = self.discovery_started.get_or_insert_with(Instant::now);
        let elapsed = started.elapsed().as_secs_f64();
        let complete = update.completed_files >= update.total_files;
        let completed = update.completed_files as u64;
        let total = update.total_files as u64;
        let percent = progress_percent(completed, total);
        let bar = progress_bar(completed, total);
        let eta = estimate_eta(completed, total, elapsed);
        self.progress_line(
            format!(
                "colic: source   [{bar}] {percent:5.1}%  {}/{} files  {} ({})  ETA {eta}",
                update.completed_files,
                update.total_files,
                update
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                human_bytes(update.bytes_hashed),
            ),
            complete,
        );
    }
}

fn progress_percent(completed: u64, total: u64) -> f64 {
    if total == 0 {
        return 100.0;
    }
    100.0 * completed.min(total) as f64 / total as f64
}

fn progress_bar(completed: u64, total: u64) -> String {
    if total == 0 {
        return "=".repeat(PROGRESS_BAR_WIDTH);
    }
    let completed = completed.min(total);
    let filled = ((completed as u128 * PROGRESS_BAR_WIDTH as u128) / total as u128) as usize;
    if filled >= PROGRESS_BAR_WIDTH {
        return "=".repeat(PROGRESS_BAR_WIDTH);
    }
    let mut bar = String::with_capacity(PROGRESS_BAR_WIDTH);
    bar.push_str(&"=".repeat(filled));
    bar.push('>');
    bar.push_str(&"-".repeat(PROGRESS_BAR_WIDTH - filled - 1));
    bar
}

fn estimate_eta(completed: u64, total: u64, elapsed_seconds: f64) -> String {
    if total == 0 || completed >= total {
        return "0s".to_owned();
    }
    if completed == 0 || elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return "--".to_owned();
    }
    let remaining = total - completed;
    let seconds = (elapsed_seconds * remaining as f64 / completed as f64).ceil();
    if !seconds.is_finite() || seconds < 0.0 {
        return "--".to_owned();
    }
    format_duration(seconds as u64)
}

fn format_duration(seconds: u64) -> String {
    if seconds >= 3600 {
        format!("{}h {:02}m {:02}s", seconds / 3600, (seconds % 3600) / 60, seconds % 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn byte_rate(bytes: u64, elapsed_seconds: f64) -> String {
    if bytes == 0 || elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return "-- MiB/s".to_owned();
    }
    format!(
        "{:.1} MiB/s",
        bytes as f64 / elapsed_seconds / (1024.0 * 1024.0)
    )
}

fn human_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.2} MiB", bytes as f64 / (1024 * 1024) as f64)
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("colic: {error}");
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
}

fn run() -> colic::Result<()> {
    match cli::parse(std::env::args().skip(1))? {
        Command::Help => {
            println!("{USAGE}");
            Ok(())
        }
        Command::InspectSource { source } => {
            eprintln!("colic: source discovery...");
            let mut progress = ConsoleProgress::new();
            let inventory = colic::source::discover_with_progress(&source, &mut |update| {
                progress.source_file(&update);
            })?;
            println!("source={}", inventory.root.display());
            println!("files={}", inventory.files.len());
            println!("tensors={}", inventory.tensors.len());
            println!("source_stored_bytes={}", inventory.source_stored_bytes);
            println!("dtype_counts={:?}", inventory.dtype_counts);
            println!("source_fingerprint={}", inventory.source_fingerprint);
            if let Some(architecture_hint) = &inventory.architecture_hint {
                println!("architecture_hint={architecture_hint}");
            }
            if let Some(config_fingerprint) = &inventory.config_fingerprint {
                println!("config_fingerprint={config_fingerprint}");
            }
            if let Some(model) = pipeline::build_semantic_ir(&inventory)? {
                println!("semantic_architecture=deepseek_v4");
                println!("semantic_layers={}", model.geometry.layers);
                println!("semantic_routed_experts={}", model.routed_experts.len());
                println!(
                    "semantic_static_layers={}",
                    model.layer_static_tensors.len()
                );
                println!("semantic_resident_tensors={}", model.resident_tensors.len());
            }
            Ok(())
        }
        Command::Verify { package } => {
            eprintln!("colic: verification...");
            let mut progress = ConsoleProgress::new();
            progress.verification_started = Some(Instant::now());
            let summary = colic::verify::verify_package_with_progress(&package, &mut |update| {
                progress.verification(update);
            })?;
            colic::verify_target::verify_target_layouts(&package)?;
            println!("package={}", package.display());
            println!("shards={}", summary.shards);
            println!("records={}", summary.records);
            Ok(())
        }
        Command::Compile(request) if request.dry_run => {
            let summary = pipeline::dry_run(&request)?;
            println!("target={}", summary.target_name);
            println!("source_tensors={}", summary.source_tensors);
            println!("source_stored_bytes={}", summary.source_stored_bytes);
            println!("projected_record_count={}", summary.plan.records.len());
            println!("projected_shard_count={}", summary.plan.shards);
            println!(
                "projected_stored_bytes={}",
                summary.plan.projected_stored_bytes
            );
            println!(
                "projected_padding_bytes={}",
                summary.plan.projected_padding_bytes
            );
            Ok(())
        }
        Command::Compile(request) => pipeline::compile(&request, &mut ConsoleProgress::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_bar_has_stable_width_and_finishes_cleanly() {
        assert_eq!(progress_bar(0, 100).len(), PROGRESS_BAR_WIDTH);
        assert_eq!(progress_bar(50, 100).len(), PROGRESS_BAR_WIDTH);
        assert_eq!(progress_bar(100, 100), "=".repeat(PROGRESS_BAR_WIDTH));
        assert_eq!(progress_bar(200, 100), "=".repeat(PROGRESS_BAR_WIDTH));
    }

    #[test]
    fn eta_formats_human_scale_time() {
        assert_eq!(estimate_eta(0, 100, 10.0), "--");
        assert_eq!(estimate_eta(50, 100, 65.0), "1m 05s");
        assert_eq!(estimate_eta(25, 100, 1225.0), "1h 01m 15s");
        assert_eq!(estimate_eta(100, 100, 10.0), "0s");
    }

    #[test]
    fn byte_rate_handles_unavailable_and_known_rates() {
        assert_eq!(byte_rate(0, 1.0), "-- MiB/s");
        assert_eq!(byte_rate(10 * 1024 * 1024, 2.0), "5.0 MiB/s");
    }
}
