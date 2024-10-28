import cv2
import os
import argparse

def create_video_from_images(
    input_dir, 
    output_file="output.mp4", 
    fps=30, 
    progressive_skip=True,
    initial_skip=1
):
    """
    Create a video from a sequence of images.
    
    Args:
        input_dir (str): Directory containing the images
        output_file (str): Output video file path
        fps (int): Frames per second
        progressive_skip (bool): Whether to progressively increase frame skipping
        initial_skip (int): Initial number of frames to skip
    """
    # Get all progress images sorted by number
    image_files = []
    for f in os.listdir(input_dir):
        if f.startswith("progress_") and f.endswith(".png"):
            # Extract number from progress_X.png
            num = int(f.split("_")[1].split(".")[0])
            image_files.append((num, f))
    
    # Sort by number
    image_files.sort()
    image_files = [f[1] for f in image_files]
    
    if not image_files:
        print("No progress images found!")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(input_dir, image_files[0]))
    height, width = first_image.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Calculate frame selection based on skip settings
    if progressive_skip:
        skip = initial_skip
        selected_indices = []
        i = 0
        while i < len(image_files):
            selected_indices.append(i)
            i += skip
            skip *= 2  # Double the skip for next iteration
    else:
        selected_indices = range(0, len(image_files), initial_skip)
    
    # Write frames
    total_frames = len(selected_indices)
    for idx, i in enumerate(selected_indices):
        if i >= len(image_files):
            break
            
        # Read image
        image_path = os.path.join(input_dir, image_files[i])
        frame = cv2.imread(image_path)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        print(f"Processing frame {idx+1}/{total_frames} (Image {i+1}/{len(image_files)})", end='\r')
    
    print("\nVideo creation completed!")
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Create video from progress images')
    parser.add_argument('input_dir', type=str, help='Directory containing progress images')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--progressive-skip', action='store_true', 
                       help='Progressively increase frame skipping')
    parser.add_argument('--initial-skip', type=int, default=1, 
                       help='Initial number of frames to skip')
    
    args = parser.parse_args()
    
    create_video_from_images(
        args.input_dir,
        args.output,
        args.fps,
        args.progressive_skip,
        args.initial_skip
    )

if __name__ == "__main__":
    main()
