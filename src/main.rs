use bevy::prelude::Image;
use bevy::{
    asset::{Handle, RenderAssetUsages},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    time::Time,
};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
// Added for finding external unrar executable
use std::fmt;
use std::fs;
use unrar::{error::UnrarError, Archive, ListSplit};

// Constants
const MAX_RAR_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100 MB
const PROCESSING_COOLDOWN: f64 = 1.0; // 1 second cooldown between processing files

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Update, file_drag_and_drop_system)
        .add_systems(Update, status_update_system)
        .add_systems(Update, display_image_system)
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

// Component to mark the image display entity
#[derive(Component)]
struct DisplayImage;

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
                    )
                    .as_str();
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
    ExtractionError(String),
}

impl fmt::Display for RarProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RarProcessingError::OpenError(e) => write!(f, "Failed to open RAR file: {}", e),
            RarProcessingError::HeaderError(e) => write!(f, "Failed to read RAR header: {}", e),
            RarProcessingError::EncodingError => write!(f, "Failed to decode filename encoding"),
            RarProcessingError::FileTooLarge(size) => write!(
                f,
                "RAR file is too large ({} MB). Maximum allowed size is {} MB",
                size / (1024 * 1024),
                MAX_RAR_FILE_SIZE / (1024 * 1024)
            ),
            RarProcessingError::NoImagesFound => {
                write!(f, "No image files found in the RAR archive")
            }
            RarProcessingError::ExtractionError(msg) => {
                write!(f, "Failed to extract image: {}", msg)
            }
        }
    }
}

impl From<UnrarError> for RarProcessingError {
    fn from(error: UnrarError) -> Self {
        RarProcessingError::OpenError(error)
    }
}

// Resource to store state about processed RAR files and images
#[derive(Resource)]
struct RarImageState {
    current_rar_path: Option<PathBuf>,
    found_images: Vec<String>,
    status: RarProcessingStatus,
    last_processed: f64, // Time when the last file was processed
    // New fields for image display
    first_image_data: Option<Vec<u8>>,
    texture_handle: Option<Handle<Image>>,
    image_loaded: bool,
}

impl Default for RarImageState {
    fn default() -> Self {
        Self {
            current_rar_path: None,
            found_images: Vec::new(),
            status: RarProcessingStatus::default(),
            last_processed: 0.0,
            first_image_data: None,
            texture_handle: None,
            image_loaded: false,
        }
    }
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
            FileDragAndDrop::DroppedFile {
                window: _,
                path_buf,
            } => {
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
                    rar_state.first_image_data = None;
                    rar_state.image_loaded = false;
                    rar_state.last_processed = current_time;

                    // Process the RAR file
                    match process_rar_file(path_buf) {
                        Ok((images, image_data)) => {
                            info!("Found {} images in RAR file", images.len());
                            for image in &images {
                                info!("Image: {}", image);
                            }

                            // Update the state with success
                            rar_state.found_images = images;
                            rar_state.first_image_data = Some(image_data);
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

// Process a RAR file and return a list of image file paths and the data of the first image
fn process_rar_file(path: &Path) -> Result<(Vec<String>, Vec<u8>), RarProcessingError> {
    // Check file size before processing
    let metadata = fs::metadata(path).map_err(|e| {
        error!("Failed to get file metadata: {}", e);
        RarProcessingError::OpenError(UnrarError {
            code: unrar::error::Code::BadData,
            when: unrar::error::When::Open,
        })
    })?;

    let file_size = metadata.len();
    if file_size > MAX_RAR_FILE_SIZE {
        return Err(RarProcessingError::FileTooLarge(file_size));
    }

    // List of common image extensions to check for
    let image_extensions: HashSet<&str> = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]
        .iter()
        .cloned()
        .collect();

    // Process the RAR file and extract image filenames
    let mut images = Vec::new();
    let mut first_image_name = None;
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
                                images.push(filename_str.clone());

                                // Remember the first image name for extraction
                                if first_image_name.is_none() {
                                    first_image_name = Some(filename_str);
                                }
                            }
                        }
                    }
                    Err(err) => {
                        warn!("Error reading RAR entry: {}", err);
                        return Err(RarProcessingError::HeaderError(err));
                    }
                }
            }
        }
        Err(e) => {
            // the error we passed in is always None
            // if the archive could not be read at all
            error!("Error opening archive: {}", e);
            return Err(RarProcessingError::OpenError(e));
        }
    }

    // Check if we found any images
    if images.is_empty() {
        warn!("No images found in the RAR file");
        return Err(RarProcessingError::NoImagesFound);
    }

    // Extract data from the first image
    let first_image = first_image_name.as_ref().unwrap().clone(); // Use as_ref() to avoid moving from the Option
    info!("Extracting first image: {}", first_image);

    // Open the archive for extraction
    let mut image_data = Vec::new();
    let mut extraction_error = None;

    // Use a separate pass to find and extract the first image
    // We'll use the same break_open approach that worked for listing
    let mut possible_extract_error = None;
    match Archive::new(path).break_open::<ListSplit>(Some(&mut possible_extract_error)) {
        Ok(archive) => {
            if let Some(error) = possible_extract_error {
                warn!("Partial RAR error during extraction: {}, continuing", error);
            }
            for entry_result in archive {
                match entry_result {
                    Ok(entry) => {
                        // Check if this is the file we want to extract
                        if let Some(filename) = entry.filename.to_str() {
                            if filename == first_image {
                                // Found our target file, extract its data
                                info!("Found target file for extraction: {}", filename);
                                // Extract the file directly from this entry
                                // We need to create a new archive to extract the data
                                // We'll use an approach that works with the current unrar crate
                                // For unrar 0.5.8, we need to manually extract the data by reading the file content
                                info!("Attempting to read file data directly");

                                // In unrar 0.5.8, we can't directly get file data from a ListSplit entry
                                // We need to create a new extraction archive
                                let mut success = false;

                                // Create a temporary file for extraction
                                let temp_dir = tempfile::tempdir()
                                    .map_err(|e| {
                                        let error_msg =
                                            format!("Failed to create temp directory: {}", e);
                                        error!("{}", &error_msg);
                                        extraction_error = Some(error_msg);
                                    })
                                    .ok();
                                if let Some(temp_dir) = temp_dir {
                                    // Try using the command-line unrar tool if available
                                    if let Some(cmd_path) = which::which("unrar").ok() {
                                        info!("Attempting to extract using command-line unrar");

                                        // Use command-line unrar to extract the file
                                        match std::process::Command::new(&cmd_path)
                                            .args(&[
                                                "e",
                                                "-o+",
                                                "-idq",
                                                path.to_str().unwrap_or(""),
                                                &filename,
                                                &format!(
                                                    "-o{}",
                                                    temp_dir.path().to_str().unwrap_or("")
                                                ),
                                            ])
                                            .status()
                                        {
                                            Ok(status) if status.success() => {
                                                // Extraction was successful, the file should be in the temp directory
                                                // The filename may be different, so we need to find it
                                                let extracted_file = match filename
                                                    .rsplit('/')
                                                    .next()
                                                    .or_else(|| filename.rsplit('\\').next())
                                                {
                                                    Some(name) => name,
                                                    None => &filename,
                                                };

                                                // Construct path to the extracted file
                                                let extracted_path =
                                                    temp_dir.path().join(extracted_file);

                                                // Read the file data
                                                match fs::read(&extracted_path) {
                                                    Ok(data) => {
                                                        info!("Successfully read {} bytes from extracted file", data.len());
                                                        image_data = data;
                                                        success = true;
                                                    }
                                                    Err(err) => {
                                                        let error_msg = format!(
                                                            "Failed to read extracted file: {}",
                                                            err
                                                        );
                                                        error!("{}", &error_msg);
                                                        extraction_error = Some(error_msg);
                                                    }
                                                }
                                            }
                                            Ok(_) => {
                                                let error_msg =
                                                    "Command-line unrar extraction failed"
                                                        .to_string();
                                                error!("{}", &error_msg);
                                                extraction_error = Some(error_msg);
                                            }
                                            Err(err) => {
                                                let error_msg = format!(
                                                    "Failed to execute command-line unrar: {}",
                                                    err
                                                );
                                                error!("{}", &error_msg);
                                                extraction_error = Some(error_msg);
                                            }
                                        }
                                    } else {
                                        // Fallback: try an alternative method if unrar isn't available
                                        let error_msg = "Command-line unrar not available, no fallback extraction method available".to_string();
                                        error!("{}", &error_msg);
                                        extraction_error = Some(error_msg);
                                    }
                                }

                                // If we failed to extract the data, try to use a simpler approach
                                if !success {
                                    let error_msg =
                                        "Failed to extract image using all available methods"
                                            .to_string();
                                    error!("{}", &error_msg);
                                    extraction_error = Some(error_msg);
                                }

                                break; // Stop after processing the first image
                            }
                        }
                    }
                    Err(err) => {
                        let error_msg = format!("Error processing entry: {}", err);
                        error!("{}", &error_msg);
                        extraction_error = Some(error_msg);
                        break;
                    }
                }
            }
        }
        Err(err) => {
            let error_msg = format!("Failed to open archive for extraction: {}", err);
            error!("{}", &error_msg);
            return Err(RarProcessingError::OpenError(err));
        }
    }

    // Check if there was an error during extraction
    if let Some(error) = extraction_error {
        return Err(RarProcessingError::ExtractionError(error));
    }

    // Verify we got image data
    if image_data.is_empty() {
        // One last attempt if all other methods failed
        if let Some(first_image) = &first_image_name {
            error!("All extraction methods failed, trying one last approach");

            // Try to use the external unrar command as a last resort
            if let Some(unrar_cmd) = which::which("unrar").ok() {
                let temp_dir = tempfile::tempdir().ok();

                if let Some(temp_dir) = temp_dir {
                    // Run unrar command with e (extract) option
                    let status = std::process::Command::new(unrar_cmd)
                        .arg("e") // extract
                        .arg("-o+") // overwrite
                        .arg("-idq") // quiet mode
                        .arg(path) // archive path
                        .arg(first_image) // file to extract
                        .arg(format!("-o{}", temp_dir.path().to_str().unwrap_or(""))) // output dir
                        .status();

                    if let Ok(status) = status {
                        if status.success() {
                            // Find the extracted file
                            let file_name = Path::new(first_image).file_name().unwrap_or_default();
                            let extracted_path = temp_dir.path().join(file_name);

                            if let Ok(data) = fs::read(&extracted_path) {
                                image_data = data;
                                info!(
                                    "Last resort extraction successful: {} bytes",
                                    image_data.len()
                                );
                            }
                        }
                    }
                }
            }
        }

        // If still empty after last attempt, return error
        if image_data.is_empty() {
            return Err(RarProcessingError::ExtractionError(
                "Failed to extract image data (empty result)".to_string(),
            ));
        }
    }
    Ok((images, image_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
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
            assert!(
                is_rar_file(&path),
                "Path {} should be recognized as a RAR file",
                path.display()
            );
        }

        // Test with invalid extensions
        let invalid_paths = vec![
            PathBuf::from("test.zip"),
            PathBuf::from("test.txt"),
            PathBuf::from("test"),
            PathBuf::from("test.rar.txt"),
        ];

        for path in invalid_paths {
            assert!(
                !is_rar_file(&path),
                "Path {} should not be recognized as a RAR file",
                path.display()
            );
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
            }
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
            (
                RarProcessingStatus::Success,
                "Successfully processed RAR file",
            ),
            (
                RarProcessingStatus::Error("Test error".to_string()),
                "Error: Test error",
            ),
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
        assert!(
            cbr_path.exists(),
            "Test file does not exist: {}",
            cbr_path.display()
        );

        // Process the CBR file
        let result = process_rar_file(&cbr_path);

        // Verify successful processing
        match result {
            Ok((image_files, image_data)) => {
                // Verify we received a non-empty vector of image files
                assert!(
                    !image_files.is_empty(),
                    "Expected non-empty vector of image files"
                );

                // Verify all returned paths have valid image extensions
                let valid_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif"];

                for image_path in &image_files {
                    let path = Path::new(image_path);
                    let extension = path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext.to_lowercase());

                    assert!(
                        extension.is_some()
                            && valid_extensions.contains(&extension.unwrap().as_str()),
                        "File has invalid image extension: {}",
                        image_path
                    );
                }

                // Verify we got image data
                assert!(!image_data.is_empty(), "Expected non-empty image data");

                println!(
                    "Successfully processed CBR file with {} images and {} bytes of image data",
                    image_files.len(),
                    image_data.len()
                );
            }
            Err(err) => {
                panic!("Failed to process CBR file: {:?}", err);
            }
        }
    }
}

// System to display the first image from the RAR file
fn display_image_system(
    mut commands: Commands,
    mut rar_state: ResMut<RarImageState>,
    mut images: ResMut<Assets<Image>>,
    image_query: Query<Entity, With<DisplayImage>>,
) {
    // Check if we have image data available and it's not yet loaded
    if let Some(image_data) = &rar_state.first_image_data {
        if !rar_state.image_loaded {
            info!(
                "Loading image data ({} bytes) into texture",
                image_data.len()
            );

            // Attempt to load the image data
            match image::load_from_memory(image_data) {
                Ok(img) => {
                    // Convert the image to RGBA
                    let rgba_img = img.to_rgba8();
                    let width = rgba_img.width();
                    let height = rgba_img.height();

                    info!("Successfully loaded image: {}x{}", width, height);

                    // Create a Bevy Image from the raw bytes
                    let size = Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    };

                    let texture = Image::new_fill(
                        size,
                        TextureDimension::D2,
                        &rgba_img.into_raw(),
                        TextureFormat::Rgba8Unorm,
                        RenderAssetUsages::RENDER_WORLD,
                    );

                    // Store the texture handle
                    let texture_handle = images.add(texture);
                    rar_state.texture_handle = Some(texture_handle.clone());

                    // Clean up any existing display sprite
                    for entity in image_query.iter() {
                        commands.entity(entity).despawn();
                    }

                    // Calculate a scale to fit the image nicely in the window
                    // Calculate a scale to fit the image nicely in the window
                    // This is a simple heuristic, could be improved based on window size
                    let max_dimension = f32::max(width as f32, height as f32);
                    let scale_factor = 800.0 / max_dimension; // Scale to fit in a 800px area

                    // Clean up any existing display sprites first
                    for entity in image_query.iter() {
                        commands.entity(entity).despawn();
                    }

                    // Create a new sprite with our image using SpriteBundle
                    commands.spawn((
                        Sprite {
                            image: texture_handle,
                            custom_size: Some(Vec2::new(
                                width as f32 * scale_factor,
                                height as f32 * scale_factor,
                            )),
                            ..default()
                        },
                        DisplayImage,
                    ));

                    // Update state
                    rar_state.image_loaded = true;

                    info!("Image displayed successfully");
                }
                Err(e) => {
                    error!("Failed to load image data: {}", e);
                    rar_state.status =
                        RarProcessingStatus::Error(format!("Failed to load image: {}", e));
                    rar_state.first_image_data = None; // Clear the data so we don't keep trying
                }
            }
        }
    } else {
        // If we don't have image data but previously had an image, clean it up
        if rar_state.image_loaded {
            for entity in image_query.iter() {
                commands.entity(entity).despawn();
            }
            rar_state.image_loaded = false;
            rar_state.texture_handle = None;
        }
    }
}
