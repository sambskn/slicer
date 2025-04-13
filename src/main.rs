use bevy::input::mouse::{MouseButtonInput, MouseWheel};
use bevy::input::ButtonState;
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
use unrar::{error::UnrarError, Archive};

// Constants
const MAX_RAR_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100 MB
const PROCESSING_COOLDOWN: f64 = 1.0; // 1 second cooldown between processing files
const MIN_BUFFER: f32 = 5.0;
const MAX_BUFFER: f32 = 50.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Update, file_drag_and_drop_system)
        .add_systems(Update, status_update_system)
        .add_systems(Update, display_image_system)
        .add_systems(Update, keyboard_navigation_system)
        .add_systems(Update, mouse_move_listen)
        .add_systems(Update, mouse_scroll_listen)
        .add_systems(Update, mouse_click_listen)
        .add_systems(Startup, setup)
        .insert_resource(RarImageState::default())
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d::default());
    commands.insert_resource(HighlightBuffer(MIN_BUFFER));
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

// Component to mark the image display entity
#[derive(Component)]
struct OverlayRect {
    is_left: bool,
}

#[derive(Resource)]
struct HighlightBuffer(f32);

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

fn mouse_move_listen(
    mut mouse_motion_events: EventReader<CursorMoved>,
    mut overlay_rect_query: Query<(&OverlayRect, &mut Transform), Without<DisplayImage>>,
    highlight_buffer: Res<HighlightBuffer>,
    sprite_query: Query<&Sprite, With<DisplayImage>>,
    sprite_transform_query: Query<&Transform, With<DisplayImage>>,
    window_query: Query<&Window>,
) {
    let mut window_size = Vec2::new(0.0, 0.0);
    let mut image_size = Vec2::new(0.0, 0.0);
    let mut image_pos = Vec2::new(0.0, 0.0);
    for sprite in &sprite_query {
        match sprite.custom_size {
            Some(size) => image_size = size,
            None => {}
        };
    }
    for transform in &sprite_transform_query {
        image_pos = transform.translation.xy();
    }
    for window in &window_query {
        window_size = Vec2::new(window.width(), window.height());
    }
    let image_left_edge = image_pos.x - (image_size.x * 0.5) + (window_size.x / 2.0);
    let image_right_edge = image_pos.x + (image_size.x * 0.5) + (window_size.x / 2.0);
    for ev in mouse_motion_events.read() {
        let mouse_pos = ev.position;
        for (rect, mut transform) in &mut overlay_rect_query {
            let rect_width = if mouse_pos.x < image_right_edge && mouse_pos.x > image_left_edge {
                (if rect.is_left {
                    mouse_pos.x - image_left_edge
                } else {
                    image_right_edge - mouse_pos.x
                }) - highlight_buffer.0
            } else {
                0.0
            };
            transform.scale.x = rect_width;
            transform.translation.x = if rect.is_left {
                (image_left_edge + (rect_width / 2.0)) - (window_size.x / 2.0)
            } else {
                (image_right_edge - (rect_width / 2.0)) - (window_size.x / 2.0)
            };
        }
    }
}

fn mouse_scroll_listen(
    mut mouse_wheel_events: EventReader<MouseWheel>,
    mut highlight_buffer: ResMut<HighlightBuffer>,
) {
    for event in mouse_wheel_events.read() {
        match event.y {
            0.0..=1.0 => {
                highlight_buffer.0 = if highlight_buffer.0 == MAX_BUFFER {
                    MAX_BUFFER
                } else {
                    highlight_buffer.0 + 1.0
                };
            }
            -1.0..0.0 => {
                highlight_buffer.0 = if highlight_buffer.0 == MIN_BUFFER {
                    MIN_BUFFER
                } else {
                    highlight_buffer.0 - 1.0
                };
            }
            _ => {}
        }
    }
}

fn mouse_click_listen(mut mouse_button_input_events: EventReader<MouseButtonInput>) {
    for event in mouse_button_input_events.read() {
        if event.button == MouseButton::Left && event.state == ButtonState::Released {
            info!("click up");
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
    FileTooLarge(u64),
    NoImagesFound,
    ExtractionError(String),
}

impl fmt::Display for RarProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RarProcessingError::OpenError(e) => write!(f, "Failed to open RAR file: {}", e),
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
    // Image display fields
    current_image_data: Option<Vec<u8>>, // Data for the currently displayed image
    current_image_index: usize,          // Index of the currently displayed image
    texture_handle: Option<Handle<Image>>,
    image_loaded: bool,
    needs_reload: bool, // Flag to indicate the image needs to be reloaded
}

impl Default for RarImageState {
    fn default() -> Self {
        Self {
            current_rar_path: None,
            found_images: Vec::new(),
            status: RarProcessingStatus::default(),
            last_processed: 0.0,
            current_image_data: None,
            current_image_index: 0,
            texture_handle: None,
            image_loaded: false,
            needs_reload: false,
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
                            rar_state.current_image_data = Some(image_data);
                            rar_state.current_image_index = 0;
                            rar_state.needs_reload = true;
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
    let mut first_image_data = Vec::new();

    // Read through the archive header by header
    let mut archive = Archive::new(path).open_for_processing().unwrap();
    while let Some(header) = archive.read_header()? {
        let filename_path = header.entry().filename.clone();
        let filename_ext_opt = filename_path.extension();
        match filename_ext_opt {
            Some(os_str) => {
                let filename_ext = os_str.to_ascii_lowercase();
                archive = if image_extensions.contains(filename_ext.to_str().unwrap()) {
                    let filename = filename_path.as_os_str().to_str().unwrap().to_string();
                    images.push(filename.to_string());
                    if first_image_data.is_empty() {
                        info!("Extracting first image: {}", filename);
                        let (data, rest) = header.read()?;
                        first_image_data = data;
                        rest
                    } else {
                        header.skip()?
                    }
                } else {
                    header.skip()?
                }
            }
            None => archive = header.skip()?,
        }
    }

    // Check if we found any images
    if images.is_empty() {
        warn!("No images found in the RAR file");
        return Err(RarProcessingError::NoImagesFound);
    }

    // Verify we got image data
    if first_image_data.is_empty() {
        return Err(RarProcessingError::ExtractionError(
            "Failed to extract image data (empty result)".to_string(),
        ));
    }

    Ok((images, first_image_data))
}

// System to display the first image from the RAR file
fn display_image_system(
    mut commands: Commands,
    mut rar_state: ResMut<RarImageState>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    image_query: Query<Entity, With<DisplayImage>>,
    overlay_query: Query<Entity, With<OverlayRect>>,
) {
    // Check if we have image data available and it's not yet loaded
    if let Some(image_data) = &rar_state.current_image_data {
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
                    for entity in overlay_query.iter() {
                        commands.entity(entity).despawn();
                    }

                    // Calculate a scale to fit the image nicely in the window
                    // This is a simple heuristic, could be improved based on window size
                    let max_dimension = f32::max(width as f32, height as f32);
                    let scale_factor = 800.0 / max_dimension; // Scale to fit in a 800px area

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

                    // Add overlay rectangles
                    let material = materials.add(Color::linear_rgba(0.0, 0.0, 0.0, 0.7));
                    commands.spawn((
                        Mesh2d(meshes.add(Rectangle::new(1.0, height as f32 * scale_factor))),
                        MeshMaterial2d(material.clone()),
                        Transform::from_xyz(-(width as f32 * scale_factor * 0.5), 0.0, 1.0),
                        OverlayRect { is_left: true },
                    ));
                    commands.spawn((
                        Mesh2d(meshes.add(Rectangle::new(1.0, height as f32 * scale_factor))),
                        MeshMaterial2d(material.clone()),
                        Transform::from_xyz(width as f32 * scale_factor * 0.5, 0.0, 1.0),
                        OverlayRect { is_left: false },
                    ));

                    // Update state
                    rar_state.image_loaded = true;

                    info!("Image displayed successfully");
                }
                Err(e) => {
                    error!("Failed to load image data: {}", e);
                    rar_state.status =
                        RarProcessingStatus::Error(format!("Failed to load image: {}", e));
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

// System to handle keyboard input for navigating between images
fn keyboard_navigation_system(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut rar_state: ResMut<RarImageState>,
) {
    // Only process keyboard input if we have images loaded
    if rar_state.found_images.is_empty() {
        return;
    }

    let image_count = rar_state.found_images.len();
    let mut new_index = rar_state.current_image_index;
    let mut changed = false;

    // Check for right arrow key (next image)
    if keyboard_input.just_pressed(KeyCode::ArrowRight) {
        new_index = (rar_state.current_image_index + 1) % image_count;
        changed = true;
    }

    // Check for left arrow key (previous image)
    if keyboard_input.just_pressed(KeyCode::ArrowLeft) {
        new_index = if rar_state.current_image_index == 0 {
            image_count - 1
        } else {
            rar_state.current_image_index - 1
        };
        changed = true;
    }

    // If the index changed, update the state and trigger a reload
    if changed && new_index != rar_state.current_image_index {
        info!("Navigating to image {}/{}", new_index + 1, image_count);
        rar_state.current_image_index = new_index;

        // Load the new image data
        if let Some(path) = &rar_state.current_rar_path {
            match load_image_by_index(path, new_index, &rar_state.found_images) {
                Ok(image_data) => {
                    rar_state.current_image_data = Some(image_data);
                    rar_state.image_loaded = false;
                    rar_state.needs_reload = true;
                }
                Err(e) => {
                    error!("Failed to load image {}: {}", new_index, e);
                    rar_state.status = RarProcessingStatus::Error(format!(
                        "Failed to load image {}: {}",
                        new_index, e
                    ));
                }
            }
        }
    }
}

// Helper function to load a specific image from the RAR file by index
fn load_image_by_index(
    path: &Path,
    index: usize,
    image_files: &[String],
) -> Result<Vec<u8>, RarProcessingError> {
    if index >= image_files.len() {
        return Err(RarProcessingError::ExtractionError(format!(
            "Image index {} out of bounds (max: {})",
            index,
            image_files.len() - 1
        )));
    }

    let target_filename = &image_files[index];
    info!("Loading image at index {}: {}", index, target_filename);

    // Read through the archive header by header
    let mut archive = Archive::new(path).open_for_processing()?;
    let mut image_data = Vec::new();

    while let Some(header) = archive.read_header()? {
        let filename = header
            .entry()
            .filename
            .as_os_str()
            .to_str()
            .unwrap_or_default();

        if filename == target_filename {
            let (data, _) = header.read()?;
            image_data = data;
            break;
        } else {
            archive = header.skip()?;
        }
    }

    if image_data.is_empty() {
        return Err(RarProcessingError::ExtractionError(format!(
            "Failed to find or extract image: {}",
            target_filename
        )));
    }

    Ok(image_data)
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
