use std::{fmt, io, path::PathBuf};

pub type Result<T> = std::result::Result<T, ColicError>;

#[derive(Debug)]
pub enum ColicError {
    Usage(String),
    SourceNotFound(PathBuf),
    InvalidSource { path: PathBuf, detail: String },
    Unsupported { stage: &'static str, detail: String },
    Io { path: PathBuf, source: io::Error },
}

impl ColicError {
    pub fn unsupported(stage: &'static str, detail: impl Into<String>) -> Self {
        Self::Unsupported {
            stage,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for ColicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) => write!(f, "usage error: {message}"),
            Self::SourceNotFound(path) => {
                write!(f, "source model does not exist: {}", path.display())
            }
            Self::InvalidSource { path, detail } => {
                write!(f, "invalid source {}: {detail}", path.display())
            }
            Self::Unsupported { stage, detail } => write!(f, "{stage}: {detail}"),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for ColicError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<colibri_format::FormatError> for ColicError {
    fn from(error: colibri_format::FormatError) -> Self {
        match error {
            colibri_format::FormatError::Invalid(message) => ColicError::Usage(message),
            colibri_format::FormatError::Io { path, source } => ColicError::Io { path, source },
        }
    }
}
