use std::path::PathBuf;

use crate::{
    error::{ColicError, Result},
    pipeline::{CodecRequest, CompileRequest, OptimizationProfile, QuantRequest, TargetRequest},
};

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    InspectSource { source: PathBuf },
    Verify { package: PathBuf },
    Compile(CompileRequest),
    Run { package: std::path::PathBuf, prompt: String, max_new: usize },
    Help,
}

pub const USAGE: &str = "Usage:\n  coli inspect-source MODEL_DIR\n  coli verify PACKAGE_DIR\n  coli compile MODEL_DIR --target native|PROFILE --quant exact|PROFILE --codec none|auto|PROFILE --opt default|size|latency -o OUTPUT [--dry-run] [--verify] [--force]";

pub fn parse<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Ok(Command::Help);
    };
    match command.as_str() {
        "help" | "--help" | "-h" => Ok(Command::Help),
        "inspect-source" => {
            let source = args
                .next()
                .ok_or_else(|| ColicError::Usage("inspect-source requires MODEL_DIR".into()))?;
            if args.next().is_some() {
                return Err(ColicError::Usage(
                    "inspect-source accepts exactly one MODEL_DIR".into(),
                ));
            }
            Ok(Command::InspectSource {
                source: PathBuf::from(source),
            })
        }
        "compile" => parse_compile(args),
        "run" => {
            let package = std::path::PathBuf::from(
                args.next().ok_or_else(|| ColicError::Usage("run requires PACKAGE_DIR".into()))?,
            );
            let mut prompt = String::from("1 2 3 4 5");
            let mut max_new = 16;
            let mut it = args;
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--prompt" => prompt = it.next().ok_or_else(|| ColicError::Usage("--prompt needs a value".into()))?,
                    "--max-new" => max_new = it.next().ok_or_else(|| ColicError::Usage("--max-new needs a value".into()))?.parse().map_err(|_| ColicError::Usage("--max-new must be a number".into()))?,
                    other => return Err(ColicError::Usage(format!("unknown run flag {other}"))),
                }
            }
            Ok(Command::Run { package, prompt, max_new })
        }
        "verify" => {
            let package = args
                .next()
                .ok_or_else(|| ColicError::Usage("verify requires PACKAGE_DIR".into()))?;
            if args.next().is_some() {
                return Err(ColicError::Usage(
                    "verify accepts exactly one PACKAGE_DIR".into(),
                ));
            }
            Ok(Command::Verify {
                package: PathBuf::from(package),
            })
        }
        other => Err(ColicError::Usage(format!("unknown command `{other}`"))),
    }
}

fn parse_compile<I>(args: I) -> Result<Command>
where
    I: Iterator<Item = String>,
{
    let mut args = args.into_iter();
    let source = PathBuf::from(
        args.next()
            .ok_or_else(|| ColicError::Usage("compile requires MODEL_DIR".into()))?,
    );
    let mut request = CompileRequest::new(source);
    while let Some(flag) = args.next() {
        let value = |args: &mut I, flag: &str| {
            args.next()
                .ok_or_else(|| ColicError::Usage(format!("{flag} requires a value")))
        };
        match flag.as_str() {
            "--target" => request.target = TargetRequest::parse(&value(&mut args, "--target")?)?,
            "--quant" => request.quant = QuantRequest::parse(&value(&mut args, "--quant")?)?,
            "--codec" => request.codec = CodecRequest::parse(&value(&mut args, "--codec")?)?,
            "--opt" => {
                request.optimization = OptimizationProfile::parse(&value(&mut args, "--opt")?)?
            }
            "-o" | "--output" => {
                request.output = Some(PathBuf::from(value(&mut args, "--output")?))
            }
            "--dry-run" => request.dry_run = true,
            "--verify" => request.verify = true,
            "--force" => request.force = true,
            other => {
                return Err(ColicError::Usage(format!(
                    "unknown compile option `{other}`"
                )));
            }
        }
    }
    if !request.dry_run && request.output.is_none() {
        return Err(ColicError::Usage(
            "compile requires -o/--output unless --dry-run is set".into(),
        ));
    }
    Ok(Command::Compile(request))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_deterministic_compile_request() {
        let command = parse(
            [
                "compile", "fixture", "--target", "native", "--quant", "exact", "--codec", "none",
                "--opt", "latency", "-o", "out.coli", "--verify",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Compile(request) = command else {
            panic!("expected compile")
        };
        assert_eq!(request.target, TargetRequest::Native);
        assert_eq!(request.quant, QuantRequest::Exact);
        assert_eq!(request.codec, CodecRequest::None);
        assert_eq!(request.optimization, OptimizationProfile::Latency);
        assert!(request.verify);
    }

    #[test]
    fn portable_target_is_rejected() {
        assert!(
            parse(
                ["compile", "fixture", "--target", "portable-v1", "--dry-run"].map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn parses_standalone_verify_command() {
        assert_eq!(
            parse(["verify", "package.coli"].map(str::to_owned)).unwrap(),
            Command::Verify {
                package: PathBuf::from("package.coli")
            }
        );
    }
}
