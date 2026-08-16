use std::time::Instant;

use colic::{
    cli::{self, Command, USAGE},
    pipeline::{self, ProgressSink, Stage},
    source::DiscoveryProgress,
};

struct ConsoleProgress {
    emission_started: Option<Instant>,
}

impl ConsoleProgress {
    fn new() -> Self {
        Self {
            emission_started: None,
        }
    }
}

impl ProgressSink for ConsoleProgress {
    fn stage(&mut self, stage: Stage) {
        if stage == Stage::Emission {
            self.emission_started = Some(Instant::now());
        }
        eprintln!("colic: {}...", stage.as_str());
    }

    fn emission(&mut self, completed: u64, total: u64, bytes: u64, total_bytes: u64) {
        let elapsed = self
            .emission_started
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let eta = if bytes > 0 && elapsed > 0.0 {
            ((total_bytes - bytes) as f64 / (bytes as f64 / elapsed)).ceil() as u64
        } else {
            0
        };
        eprintln!(
            "colic: emission {completed}/{total} records, {}/{}; ETA {}s",
            human_bytes(bytes),
            human_bytes(total_bytes),
            eta
        );
    }

    fn source_file(&mut self, update: &DiscoveryProgress) {
        eprintln!(
            "colic: source fingerprint {}/{}: {} ({})",
            update.completed_files,
            update.total_files,
            update
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            human_bytes(update.bytes_hashed),
        );
    }
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
            let summary = colic::verify::verify_package(&package)?;
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
