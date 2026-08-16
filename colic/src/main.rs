mod cli;

use std::time::{Duration, Instant};

use cli::{Command, USAGE};
use colic::{pipeline, source::DiscoveryProgress, verify::VerificationProgress};

struct ConsoleProgress {
    last: Option<Instant>,
}

impl ConsoleProgress {
    fn new() -> Self {
        Self { last: None }
    }

    fn ready(&mut self) -> bool {
        let now = Instant::now();
        if self
            .last
            .is_none_or(|last| now.duration_since(last) >= Duration::from_millis(100))
        {
            self.last = Some(now);
            true
        } else {
            false
        }
    }

    fn source_file(&mut self, update: &DiscoveryProgress) {
        if self.ready() || update.completed_files == update.total_files {
            eprintln!(
                "colic: source [{}/{}] {} ({})",
                update.completed_files,
                update.total_files,
                update.path.display(),
                human_bytes(update.bytes_hashed),
            );
        }
    }

    fn verification(&mut self, update: VerificationProgress) {
        if self.ready() || update.completed_records == update.total_records {
            eprintln!(
                "colic: verify [{}/{} records] shard {}/{} ({})",
                update.completed_records,
                update.total_records,
                update.current_shard + 1,
                update.total_shards,
                human_bytes(update.verified_bytes),
            );
        }
    }
}

impl pipeline::ProgressSink for ConsoleProgress {
    fn event(&mut self, event: &pipeline::ProgressEvent) {
        match event {
            pipeline::ProgressEvent::Stage(stage) => eprintln!("colic: stage={stage:?}"),
            pipeline::ProgressEvent::Target(name) => eprintln!("colic: target={name}"),
            pipeline::ProgressEvent::Source(update) => self.source_file(update),
            pipeline::ProgressEvent::Record {
                completed,
                total,
                shard,
                bytes_written,
            } => {
                if self.ready() || completed == total {
                    eprintln!(
                        "colic: records [{completed}/{total}] shard={} ({})",
                        shard + 1,
                        human_bytes(*bytes_written),
                    );
                }
            }
            pipeline::ProgressEvent::Verify(update) => self.verification(*update),
        }
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
                println!(
                    "semantic_unclassified_tensors={}",
                    model.resident_tensors.len()
                );
                println!(
                    "semantic_assets={}",
                    model.assets.tokenizer.len() + model.assets.config.is_some() as usize
                );
                let mut has_fp8 = false;
                let mut has_mxfp4 = false;
                for expert in model.routed_experts.values() {
                    for matrix in [&expert.gate, &expert.up, &expert.down] {
                        match matrix.quantization.math_format {
                            colic::ir::MathFormat::Fp8E4M3 => has_fp8 = true,
                            colic::ir::MathFormat::MxFp4E2M1 => has_mxfp4 = true,
                        }
                    }
                }
                let mut formats = Vec::new();
                if has_fp8 {
                    formats.push("fp8-e4m3");
                }
                if has_mxfp4 {
                    formats.push("mxfp4-e2m1");
                }
                println!("semantic_expert_quant_formats={}", formats.join(","));
            }
            Ok(())
        }
        Command::Verify { package } => {
            eprintln!("colic: verification...");
            let mut progress = ConsoleProgress::new();
            let summary = colic::verify::verify_package_with_progress(&package, &mut |update| {
                progress.verification(update);
            })?;
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
