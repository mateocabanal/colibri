//! #201 — `colic plan` CLI surface: dry-run planning report, JSON + text.
//!
//! Plans never read tensor payloads — only config + index metadata — so
//! `colic plan` runs fast and can run without emitting model data.

use std::path::PathBuf;

use crate::{
    error::{ColicError, Result},
    pipeline,
    plan::{
        cost::Objective,
        machine::MachineProfile,
        planner::{build_plan, PlanRequest},
    },
    source,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanOptions {
    pub source: PathBuf,
    pub objective: Objective,
    pub context_tokens: u32,
    pub batch: u32,
    /// Synthetic profile file (reproducible planning without hardware).
    pub machine_profile: Option<PathBuf>,
    pub json: bool,
    /// Output file for the plan JSON (default stdout).
    pub output: Option<PathBuf>,
}

pub fn run(options: &PlanOptions) -> Result<String> {
    let machine = match &options.machine_profile {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|error| ColicError::Io {
                path: path.clone(),
                source: error,
            })?;
            let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
                ColicError::Usage(format!("machine profile {} is not valid JSON: {error}", path.display()))
            })?;
            MachineProfile::from_json(&value)?
        }
        None => MachineProfile::probe(),
    };
    let inventory = source::discover(&options.source)?;
    let model = pipeline::build_semantic_ir(&inventory)?.ok_or_else(|| {
        ColicError::unsupported(
            "semantic IR",
            "no supported architecture frontend matched this source model",
        )
    })?;
    let config = crate::source::config(&inventory.root)?;
    let request = PlanRequest {
        objective: options.objective,
        context_tokens: options.context_tokens,
        batch: options.batch,
        target_profile: None,
    };
    let plan = build_plan(
        &model,
        &machine,
        config.as_ref(),
        &request,
        Some(inventory.source_fingerprint.clone()),
    )?;
    plan.validate().map_err(|reason| ColicError::unsupported("target planning", reason))?;
    let json = plan.to_json();
    let text = if options.json {
        serde_json::to_string_pretty(&json)
            .map_err(|error| ColicError::Usage(format!("plan serialization failed: {error}")))?
    } else {
        summary_text(&plan)
    };
    if let Some(output) = &options.output {
        std::fs::write(output, &text).map_err(|error| ColicError::Io {
            path: output.clone(),
            source: error,
        })?;
    }
    Ok(text)
}

/// Human-readable one-screen summary of the plan.
fn summary_text(plan: &crate::plan::ir::PhysicalPlan) -> String {
    let mut text = String::new();
    text.push_str(&format!(
        "plan: objective={} context={} planner_v{} cost_v{} schema_v{}\n",
        plan.objective, plan.context_tokens, plan.planner_version, plan.cost_model_version, plan.plan_schema_version
    ));
    text.push_str(&plan.machine.summary_text());
    let mut totals: Vec<(String, u64, u64)> = Vec::new();
    for tensor in &plan.tensors {
        let label = format!(
            "{} {} {} {}",
            tensor.role.as_str(),
            tensor.math.as_str(),
            tensor.layout.as_str(),
            tensor.backend.as_str()
        );
        if let Some(entry) = totals.iter_mut().find(|(existing, _, _)| *existing == label) {
            entry.1 += tensor.stored_bytes;
            entry.2 += 1;
        } else {
            totals.push((label, tensor.stored_bytes, 1));
        }
    }
    text.push_str("\nrecords (family math layout backend -> bytes [count]):\n");
    for (label, bytes, count) in &totals {
        text.push_str(&format!("  {label} -> {} [{count}]\n", human_bytes(*bytes)));
    }
    let total: u64 = plan.tensors.iter().map(|tensor| tensor.stored_bytes).sum();
    text.push_str(&format!("total projected package bytes: {}\n", human_bytes(total)));
    text.push_str("\ndecisions:\n");
    for decision in &plan.decisions {
        text.push_str(&format!("  {}: {}\n", decision.subject, decision.chosen));
        for (candidate, reason) in &decision.rejected {
            text.push_str(&format!("    rejected {candidate}: {reason}\n"));
        }
    }
    text
}

fn human_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.2} MiB", bytes as f64 / (1024 * 1024) as f64)
    }
}