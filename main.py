import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math

class GoldenSpiralArtist:
    def __init__(self, target_path, num_spirals=1000, max_spiral_size=None, min_spiral_size=10):
        self.target_image = cv2.imread(target_path)
        self.target_image = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.target_image.shape[:2]
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self.num_spirals = num_spirals
        
        # Set maximum spiral size to 1/4 of the smallest image dimension if not specified
        self.max_spiral_size = max_spiral_size or min(self.height, self.width) // 2
        self.min_spiral_size = min_spiral_size
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    def get_spiral_size_for_iteration(self, iteration):
        """Calculate spiral size based on current iteration."""
        # Exponentially decrease spiral size from max to min
        progress = iteration / self.num_spirals
        # Use exponential decay for smoother size transition
        size = self.max_spiral_size * math.exp(-3*progress)
        return max(int(size), self.min_spiral_size)

    def create_golden_spiral(self, center, size, color, angle=0):
        """Create a golden spiral brush stroke."""
        # Create a smaller canvas just for the spiral region
        spiral = np.zeros((size, size, 3), dtype=np.uint8)
        a = size / 15  # Reduced scale factor to allow for more rotations
        
        # Increased rotations significantly (from 8*pi to 16*pi)
        # and increased points for smoother curves
        theta = np.linspace(0, 16*np.pi, 400)
        
        # Generate spiral points with much slower growth rate
        # Using a smaller divisor creates a longer, more gradual spiral
        r = a * np.exp(theta / (self.phi * 2.5))  # Much slower growth rate
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Rotate spiral
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        points = np.dot(np.column_stack((x, y)), rot_matrix)
        
        # Scale points to fit within the size
        max_dim = np.max(np.abs(points))
        scale_factor = (size/2 * 0.95) / max_dim  # 0.95 to leave a small margin
        points *= scale_factor
        
        # Translate to center
        points[:, 0] += center[0]
        points[:, 1] += center[1]
        
        # Draw spiral with thicker lines for larger spirals
        points = points.astype(np.int32)
        valid_points = (points[:, 0] >= 0) & (points[:, 0] < size) & \
                      (points[:, 1] >= 0) & (points[:, 1] < size)
        points = points[valid_points]
        
        # Adjust line thickness based on spiral size
        thickness = max(1, size // 80)  # Reduced thickness for finer lines
        
        for i in range(len(points)-1):
            cv2.line(spiral, 
                    (points[i][0], points[i][1]), 
                    (points[i+1][0], points[i+1][1]), 
                    color.tolist(), 
                    thickness)
        
        return spiral

    def calculate_error(self, img1, img2):
        """
        Calculate error between two images using a combination of structural and gradient-based metrics.
        Returns a weighted combination of SSIM and gradient differences.
        """
        # Convert to float
        img1 = img1.astype(float)
        img2 = img2.astype(float)
        
        # Calculate SSIM
        def ssim(img1, img2, k1=0.01, k2=0.03, win_size=3):
            # Constants
            L = 255  # Dynamic range
            c1 = (k1 * L) ** 2
            c2 = (k2 * L) ** 2
            
            # Means
            mu1 = cv2.GaussianBlur(img1, (win_size, win_size), 1.5)
            mu2 = cv2.GaussianBlur(img2, (win_size, win_size), 1.5)
            
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            
            # Variances and covariance
            sigma1_sq = cv2.GaussianBlur(img1 * img1, (win_size, win_size), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2 * img2, (win_size, win_size), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1 * img2, (win_size, win_size), 1.5) - mu1_mu2
            
            # SSIM
            ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            
            return np.mean(ssim_map)
        
        # Calculate gradients using Sobel
        def gradient_error(img1, img2):
            # X and Y gradients
            sobelx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
            sobely1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
            sobelx2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
            sobely2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            mag1 = np.sqrt(sobelx1**2 + sobely1**2)
            mag2 = np.sqrt(sobelx2**2 + sobely2**2)
            
            return np.mean((mag1 - mag2)**2)
        
        # Calculate errors for each channel
        ssim_error = 0
        gradient_error_val = 0
        
        for c in range(3):  # RGB channels
            ssim_error += (1 - ssim(img1[..., c], img2[..., c]))
            gradient_error_val += gradient_error(img1[..., c], img2[..., c])
        
        ssim_error /= 3
        gradient_error_val /= 3
        
        # Normalize gradient error
        gradient_error_val = gradient_error_val / (255 * 255)  # Normalize to [0,1] range
        
        # Weighted combination
        w_ssim = 0.7
        w_gradient = 0.3
        
        return w_ssim * ssim_error + w_gradient * gradient_error_val

    def find_best_spiral(self, region, spiral_size):
        """Find the best spiral parameters for a given region."""
        mean_color = np.mean(region, axis=(0, 1))
        center_y, center_x = region.shape[0]//2, region.shape[1]//2
        best_error = float('inf')
        best_params = None
        
        # Get region bounds
        y_start = center_y - region.shape[0]//2
        y_end = y_start + region.shape[0]
        x_start = center_x - region.shape[1]//2
        x_end = x_start + region.shape[1]
        
        # Test different angles with finer granularity
        for angle in np.linspace(0, 2*np.pi, 12):  # Increased from 8 to 12 angles
            test_spiral = self.create_golden_spiral(
                (spiral_size//2, spiral_size//2),
                spiral_size,
                mean_color,
                angle
            )
            # Crop spiral to match region size
            test_region = test_spiral[y_start:y_end, x_start:x_end]
            error = self.calculate_error(test_region, region)
            
            if error < best_error:
                best_error = error
                best_params = (mean_color, angle)
        
        return best_params

    def get_random_spiral_size(self, iteration):
        """
        Get a random spiral size that trends smaller over time while maintaining size variety.
        Uses both power law distribution and time-based reduction.
        """
        # Calculate time-based maximum size reduction
        progress = iteration / self.num_spirals
        current_max_size = self.max_spiral_size * math.exp(-6 * progress)
        current_max_size = max(int(current_max_size), self.min_spiral_size * 2)
        
        # Use power law distribution for size variety
        power = 2.0  # Adjust this to change the size distribution
        random_val = np.random.power(power)
        random_val = 1 - random_val  # Invert so larger spirals are less common
        
        # Calculate size with both random variation and time-based reduction
        size = int(self.min_spiral_size + random_val * (current_max_size - self.min_spiral_size))
        
        # Add occasional larger spirals with decreasing probability
        if np.random.random() < 0.1 * (1 - progress):  # 10% chance at start, decreasing over time
            size = int(self.min_spiral_size + random_val * (self.max_spiral_size - self.min_spiral_size))
        
        return size

    def draw(self):
        """Draw the image using golden spirals of varying sizes."""
        for i in range(self.num_spirals):
            # Get random spiral size with time-based reduction
            current_spiral_size = self.get_random_spiral_size(i)
            
            # Randomly select region
            x = np.random.randint(0, self.width - current_spiral_size)
            y = np.random.randint(0, self.height - current_spiral_size)
            
            region = self.target_image[y:y+current_spiral_size, 
                                     x:x+current_spiral_size]
            
            # Find best spiral parameters for this region
            color, angle = self.find_best_spiral(region, current_spiral_size)
            
            # Draw spiral
            spiral_patch = self.create_golden_spiral(
                (current_spiral_size//2, current_spiral_size//2),
                current_spiral_size,
                color,
                angle
            )
            
            # Copy spiral patch to canvas with alpha blending
            mask = (spiral_patch > 0).astype(float)
            
            # Adjust alpha based on spiral size and iteration
            base_alpha = 0.7  # Base opacity
            size_factor = current_spiral_size / self.max_spiral_size
            progress = i / self.num_spirals
            
            # Larger spirals are more transparent, but transparency decreases over time
            alpha = base_alpha * (1 - size_factor * 0.5 * (1 - progress))
            mask = mask * alpha
            
            self.canvas[y:y+current_spiral_size, x:x+current_spiral_size] = \
                (self.canvas[y:y+current_spiral_size, x:x+current_spiral_size] * (1 - mask) + 
                 spiral_patch * mask).astype(np.uint8)
            
            # Calculate and print error every 100 spirals
            if (i + 1) % 100 == 0:
                error = self.calculate_error(self.canvas, self.target_image)
                print(f"Iteration {i+1}, Error: {error:.2f}, Spiral Size: {current_spiral_size}")
                
            if (i + 1) % 200 == 0:
                self.save_progress(f"progress_{i+1}.png")

    def save_progress(self, filename):
        """Save current state of the canvas."""
        Image.fromarray(self.canvas).save(filename)

if __name__ == "__main__":
    # Example usage with progressive spiral sizes
    artist = GoldenSpiralArtist(
        "example.png",
        num_spirals=100000,
        max_spiral_size=None,  # Will be set to 1/4 of image size
        min_spiral_size=5
    )
    artist.draw()
