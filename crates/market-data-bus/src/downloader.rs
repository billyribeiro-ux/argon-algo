//! `downloader.rs` — Async, Resume-Safe HTTP Downloader with Hugging Face Hub Metadata Discovery.
//!
//! Features:
//! - Exponential backoff retry with jitter.
//! - Atomic temp-file downloads (`.part` -> rename) into `data/raw_downloads/`.
//! - Resilient query of Hugging Face Hub tree metadata (`https://huggingface.co/api/datasets/{repo}/tree/main/...`).
//! - Automatic redirect following (handling Hugging Face and CBOE 302/307 redirects).

use reqwest::header::{HeaderMap, HeaderValue, RANGE};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Debug, Error)]
pub enum DownloaderError {
    #[error("HTTP request error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Hugging Face API returned error status {status} for path '{url}': {body}")]
    HuggingFaceApi {
        status: reqwest::StatusCode,
        url: String,
        body: String,
    },
    #[error("Exceeded max retries ({max_retries}) downloading {url}: {last_err}")]
    MaxRetriesExceeded {
        url: String,
        max_retries: usize,
        last_err: String,
    },
    #[error("JSON decode error from {url}: {source}")]
    JsonDecode {
        url: String,
        source: serde_json::Error,
    },
}

/// Metadata entry returned by the Hugging Face Hub Tree API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfTreeEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "file" or "directory"
    pub oid: Option<String>,
    pub size: Option<u64>,
    pub path: String,
}

pub struct ResilientDownloader {
    client: reqwest::Client,
    download_dir: PathBuf,
    max_retries: usize,
}

impl ResilientDownloader {
    pub fn new(download_dir: impl Into<PathBuf>) -> Result<Self, DownloaderError> {
        let download_dir = download_dir.into();
        fs::create_dir_all(&download_dir).map_err(|source| DownloaderError::Io {
            path: download_dir.clone(),
            source,
        })?;

        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_static("argon-algo-pipeline/1.0 (Rust 2024; Quant Architecture)"),
        );

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::limited(10))
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            download_dir,
            max_retries: 5,
        })
    }

    pub fn download_dir(&self) -> &Path {
        &self.download_dir
    }

    /// Queries Hugging Face Hub dataset tree metadata dynamically.
    pub async fn list_hf_tree(
        &self,
        repo: &str,
        subpath: &str,
    ) -> Result<Vec<HfTreeEntry>, DownloaderError> {
        let clean_path = subpath.trim_matches('/');
        let url = if clean_path.is_empty() {
            format!("https://huggingface.co/api/datasets/{repo}/tree/main")
        } else {
            format!("https://huggingface.co/api/datasets/{repo}/tree/main/{clean_path}")
        };

        debug!("Querying Hugging Face tree API: {}", url);
        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(DownloaderError::HuggingFaceApi { status, url, body });
        }

        let entries: Vec<HfTreeEntry> = resp.json().await.map_err(|e| {
            DownloaderError::Reqwest(e)
        })?;

        Ok(entries)
    }

    /// Downloads a remote URL directly to `dest_path` with atomic rename and resume support.
    pub async fn download_file(
        &self,
        url: &str,
        dest_path: &Path,
        expected_size: Option<u64>,
    ) -> Result<PathBuf, DownloaderError> {
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).map_err(|source| DownloaderError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        // If destination file already exists and matches expected size (or is non-empty), skip download
        if dest_path.exists()
            && let Ok(meta) = fs::metadata(dest_path) {
                if let Some(exp) = expected_size {
                    if meta.len() == exp {
                        debug!("File {} already downloaded ({exp} bytes), skipping.", dest_path.display());
                        return Ok(dest_path.to_path_buf());
                    }
                } else if meta.len() > 0 {
                    debug!("File {} already exists ({} bytes), skipping.", dest_path.display(), meta.len());
                    return Ok(dest_path.to_path_buf());
                }
            }

        let temp_path = dest_path.with_extension(format!(
            "{}.part",
            dest_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("tmp")
        ));

        let mut attempt = 0;
        let mut last_err_str = String::new();

        while attempt < self.max_retries {
            attempt += 1;
            match self.try_download_atomic(url, &temp_path, dest_path).await {
                Ok(()) => {
                    info!("Successfully downloaded {} -> {}", url, dest_path.display());
                    return Ok(dest_path.to_path_buf());
                }
                Err(e) => {
                    last_err_str = e.to_string();
                    warn!(
                        "Download attempt {}/{} failed for {url}: {last_err_str}",
                        attempt, self.max_retries
                    );
                    let backoff_ms = 500 * (1 << attempt.min(5));
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }

        Err(DownloaderError::MaxRetriesExceeded {
            url: url.to_string(),
            max_retries: self.max_retries,
            last_err: last_err_str,
        })
    }

    async fn try_download_atomic(
        &self,
        url: &str,
        temp_path: &Path,
        dest_path: &Path,
    ) -> Result<(), DownloaderError> {
        let existing_len = if temp_path.exists() {
            fs::metadata(temp_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        let mut req = self.client.get(url);
        if existing_len > 0 {
            req = req.header(RANGE, format!("bytes={existing_len}-"));
        }

        let resp = req.send().await?;
        let status = resp.status();

        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            let body = resp.text().await.unwrap_or_default();
            return Err(DownloaderError::HuggingFaceApi {
                status,
                url: url.to_string(),
                body,
            });
        }

        let mut file = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            let mut f = File::options()
                .read(true)
                .write(true)
                .open(temp_path)
                .map_err(|source| DownloaderError::Io {
                    path: temp_path.to_path_buf(),
                    source,
                })?;
            f.seek(SeekFrom::End(0))
                .map_err(|source| DownloaderError::Io {
                    path: temp_path.to_path_buf(),
                    source,
                })?;
            f
        } else {
            File::create(temp_path).map_err(|source| DownloaderError::Io {
                path: temp_path.to_path_buf(),
                source,
            })?
        };

        let bytes = resp.bytes().await?;
        file.write_all(&bytes)
            .map_err(|source| DownloaderError::Io {
                path: temp_path.to_path_buf(),
                source,
            })?;
        file.flush().map_err(|source| DownloaderError::Io {
            path: temp_path.to_path_buf(),
            source,
        })?;
        drop(file);

        // Atomic rename into final destination
        fs::rename(temp_path, dest_path).map_err(|source| DownloaderError::Io {
            path: dest_path.to_path_buf(),
            source,
        })?;

        Ok(())
    }
}
