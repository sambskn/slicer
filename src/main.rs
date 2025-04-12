use std::path::{Path, PathBuf};
use std::collections::HashSet;
use std::ffi::OsStr;

use bevy::{prelude::*, time::Time};
use unrar::{Archive, error::UnrarError, ListSplit};
use std::fmt;
use std::fs;

// Constants
const MAX_RAR_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100 MB
const PROCESSING_COOLDOWN: f64 = 1.0; // 1 second cooldown between processing files

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Update, file_drag_and_drop_system)
        .add_systems(Update, status_update_system)
        .add_systems(Startup, setup)
        .insert_resource(RarImageState::default())
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d::default());
    
    // Status text (will be updated by the status_update_system)
    commands.spawn((
        Text::new("Status: Idle"),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        Transform::from_xyz(0.0, -50.0, 0.0),
        StatusText,
    ));
}

// Component to mark the status text entity
#[derive(Component)]
struct StatusText;

// System to update the status text based on RarImageState
fn status_update_system(
    rar_state: Res<RarImageState>,
    mut query: Query<&mut Text, With<StatusText>>,
) {
    if rar_state.is_changed() {
        if let Ok(mut text) = query.get_single_mut() {
            text.0 = format!("Status: {}", rar_state.status);
            
            // Add file info if available
            if let Some(path) = &rar_state.current_rar_path {
                if matches!(rar_state.status, RarProcessingStatus::Success) {
                    text.0 += format!(
                        "\nProcessed: {}\nFound {} images", 
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        rar_state.found_images.len()
                    ).as_str();
                }
            }
        }
    }
}

// Enum to represent different states of RAR processing
#[derive(Debug, Clone, PartialEq, Eq)]
enum RarProcessingStatus {
    Idle,
    Loading,
    Success,
    Error(String),
}

impl Default for RarProcessingStatus {
    fn default() -> Self {
        RarProcessingStatus::Idle
    }
}

impl fmt::Display for RarProcessingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RarProcessingStatus::Idle => write!(f, "Idle"),
            RarProcessingStatus::Loading => write!(f, "Loading..."),
            RarProcessingStatus::Success => write!(f, "Successfully processed RAR file"),
            RarProcessingStatus::Error(err) => write!(f, "Error: {}", err),
        }
    }
}

// Custom error type for RAR processing
#[derive(Debug)]
enum RarProcessingError {
    OpenError(UnrarError),
    HeaderError(UnrarError),
    EncodingError,
    FileTooLarge(u64),
    NoImagesFound,
}

impl fmt::Display for RarProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RarProcessingError::OpenError(e) => write!(f, "Failed to open RAR file: {}", e),
            RarProcessingError::HeaderError(e) => write!(f, "Failed to read RAR header: {}", e),
            RarProcessingError::EncodingError => write!(f, "Failed to decode filename encoding"),
            RarProcessingError::FileTooLarge(size) => write!(f, "RAR file is too large ({} MB). Maximum allowed size is {} MB", 
                size / (1024 * 1024), MAX_RAR_FILE_SIZE / (1024 * 1024)),
            RarProcessingError::NoImagesFound => write!(f, "No image files found in the RAR archive"),
        }
    }
}

impl From<UnrarError> for RarProcessingError {
    fn from(error: UnrarError) -> Self {
        RarProcessingError::OpenError(error)
    }
}

// Resource to store state about processed RAR files and images
#[derive(Resource, Default)]
struct RarImageState {
    current_rar_path: Option<PathBuf>,
    found_images: Vec<String>,
    status: RarProcessingStatus,
    last_processed: f64, // Time when the last file was processed
}

impl RarImageState {
    fn can_process(&self, current_time: f64) -> bool {
        current_time - self.last_processed >= PROCESSING_COOLDOWN
    }
}

fn file_drag_and_drop_system(
    mut events: EventReader<FileDragAndDrop>,
    mut rar_state: ResMut<RarImageState>,
    time: Res<Time>,
) {
    for event in events.read() {
        match event {
            FileDragAndDrop::DroppedFile { window: _, path_buf } => {
                info!("File dropped: {}", path_buf.display());
                
                // Check if we can process a new file (rate limiting)
                let current_time = time.elapsed_secs_f64();
                if !rar_state.can_process(current_time) {
                    info!("Processing cooldown in effect, skipping file");
                    return;
                }
                
                if is_rar_file(path_buf) {
                    // Set loading state
                    rar_state.status = RarProcessingStatus::Loading;
                    rar_state.current_rar_path = Some(path_buf.clone());
                    rar_state.found_images.clear();
                    rar_state.last_processed = current_time;
                    
                    // Process the RAR file
                    match process_rar_file(path_buf) {
                        Ok(images) => {
                            info!("Found {} images in RAR file", images.len());
                            for image in &images {
                                info!("Image: {}", image);
                            }
                            
                            // Update the state with success
                            rar_state.found_images = images;
                            rar_state.status = RarProcessingStatus::Success;
                        }
                        Err(e) => {
                            error!("Error processing RAR file: {}", e);
                            rar_state.status = RarProcessingStatus::Error(e.to_string());
                        }
                    }
                } else {
                    info!("Dropped file is not a RAR file");
                    rar_state.status = RarProcessingStatus::Error("Not a RAR file".to_string());
                }
            }
            _ => {}
        }
    }
}

// Check if the file has a .rar extension
fn is_rar_file(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("rar") || ext.eq_ignore_ascii_case("cbr"))
        .unwrap_or(false)
}

// Process a RAR file and return a list of image file paths
fn process_rar_file(path: &Path) -> Result<Vec<String>, RarProcessingError> {
    // Check file size before processing
    let metadata = fs::metadata(path).map_err(|e| {
        error!("Failed to get file metadata: {}", e);
        RarProcessingError::OpenError(UnrarError { code: unrar::error::Code::BadData, when: unrar::error::When::Open })
    })?;
    
    let file_size = metadata.len();
    if file_size > MAX_RAR_FILE_SIZE {
        return Err(RarProcessingError::FileTooLarge(file_size));
    }
    
    // List of common image extensions to check for
    let image_extensions: HashSet<&str> = [
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"
    ].iter().cloned().collect();

    // Process the RAR file and extract image filenames
    let mut images = Vec::new();
    let mut possible_error = None;
    match Archive::new(path).break_open::<ListSplit>(Some(&mut possible_error)) {
            Ok(archive) => {
                if let Some(error) = possible_error {
                    // If the error's data field holds an OpenArchive, an error occurred while opening,
                    // the archive is partly broken (e.g. broken header), but is still readable from.
                    // We're going to use the archive and list its contents, but log the warning.
                    warn!("Partial RAR error: {}, continuing with extraction", error);
                }
                for entry in archive {
                    match entry {
                        Ok(e) => {
                            // Get the filename from the entry
                            let filename = e.filename;
                            // Convert filename to string, handle encoding errors
                            let filename_str = match filename.to_str() {
                                Some(name) => name.to_string(),
                                None => {
                                    warn!("Failed to decode filename encoding");
                                    return Err(RarProcessingError::EncodingError);
                                }
                            };
                            
                            // Check if the file has an image extension
                            if let Some(extension) = Path::new(&filename_str)
                                .extension()
                                .and_then(OsStr::to_str)
                                .map(|ext| ext.to_lowercase())
                            {
                                if image_extensions.contains(extension.as_str()) {
                                    debug!("Found image: {}", filename_str);
                                    images.push(filename_str);
                                }
                            }
                        },
                        Err(err) => {
                            warn!("Error reading RAR entry: {}", err);
                            return Err(RarProcessingError::HeaderError(err));
                        },
                    }
                }    
            }
            Err(e) => {
                // the error we passed in is always None
                // if the archive could not be read at all
                println!("Error: {}", e);
            }
        }
    // Check if we found any images
    if images.is_empty() {
        warn!("No images found in the RAR file");
        return Err(RarProcessingError::NoImagesFound);
    }

    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::io::Write;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_is_rar_file() {
        // Test with valid RAR extensions
        let valid_paths = vec![
            PathBuf::from("test.rar"),
            PathBuf::from("path/to/archive.RAR"),
            PathBuf::from("/absolute/path/file.rar"),
        ];

        for path in valid_paths {
            assert!(is_rar_file(&path), "Path {} should be recognized as a RAR file", path.display());
        }

        // Test with invalid extensions
        let invalid_paths = vec![
            PathBuf::from("test.zip"),
            PathBuf::from("test.txt"),
            PathBuf::from("test"),
            PathBuf::from("test.rar.txt"),
        ];

        for path in invalid_paths {
            assert!(!is_rar_file(&path), "Path {} should not be recognized as a RAR file", path.display());
        }
    }

    #[test]
    fn test_file_size_validation() {
        // Create a temporary directory
        let dir = tempdir().expect("Failed to create temp dir");
        
        // Create a file larger than the maximum allowed size
        let file_path = dir.path().join("large.rar");
        let mut file = File::create(&file_path).expect("Failed to create test file");
        
        // Write data to the file to make it large enough
        let data = [0u8; 1024]; // 1KB chunk
        let chunks_to_write = (MAX_RAR_FILE_SIZE as usize / 1024) + 1; // One more than the max
        
        for _ in 0..chunks_to_write {
            file.write_all(&data).expect("Failed to write to test file");
        }
        
        // Test the process_rar_file function with the large file
        let result = process_rar_file(&file_path);
        
        // Verify the result is an error of type FileTooLarge
        match result {
            Err(RarProcessingError::FileTooLarge(_)) => {
                // This is the expected case
                assert!(true);
            },
            _ => {
                panic!("Expected FileTooLarge error but got: {:?}", result);
            }
        }
    }

    #[test]
    fn test_rar_processing_status_display() {
        // Test all status variants for correct string representation
        let statuses = vec![
            (RarProcessingStatus::Idle, "Idle"),
            (RarProcessingStatus::Loading, "Loading..."),
            (RarProcessingStatus::Success, "Successfully processed RAR file"),
            (RarProcessingStatus::Error("Test error".to_string()), "Error: Test error"),
        ];

        for (status, expected) in statuses {
            assert_eq!(status.to_string(), expected);
        }
    }

    #[test]
    fn test_cbr_file_processing() {
        // Test with a real CBR file (CBR is a valid subset of RAR)
        // This file is approximately 93MB, which is below the MAX_RAR_FILE_SIZE (100MB)
        let cbr_path = PathBuf::from("inputs/nintendo_power_069_feb_1995.cbr");
        
        // Make sure the test file exists
        assert!(cbr_path.exists(), "Test file does not exist: {}", cbr_path.display());
        
        // Process the CBR file
        let result = process_rar_file(&cbr_path);
        
        // Verify successful processing
        match result {
            Ok(image_files) => {
                // Verify we received a non-empty vector of image files
                assert!(!image_files.is_empty(), "Expected non-empty vector of image files");
                
                // Verify all returned paths have valid image extensions
                let valid_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif"];
                
                for image_path in &image_files {
                    let path = Path::new(image_path);
                    let extension = path.extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext.to_lowercase());
                    
                    assert!(
                        extension.is_some() && valid_extensions.contains(&extension.unwrap().as_str()),
                        "File has invalid image extension: {}", image_path
                    );
                }
                
                println!("Successfully processed CBR file with {} images", image_files.len());
            },
            Err(err) => {
                panic!("Failed to process CBR file: {:?}", err);
            }
        }
    }
}
