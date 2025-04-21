import os
import numpy as np
import random
from PIL import Image, ImageOps, ImageChops
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from config import get_custom_dataset_folder

class ImprovedHandwrittenNumbersDataset(Dataset):
    digits_per_class = 1000

    def __init__(self, custom_dataset_folder, mnist_dataset, max_digits=5, length=100_000, include_leading_zeros=False, with_symbols=True, seed=None, pre_generate=False, num_threads=1):
        self.custom_dataset_folder = custom_dataset_folder
        self.mnist_dataset = mnist_dataset
        self.max_digits = max_digits
        self.include_leading_zeros = include_leading_zeros
        self.seed = seed
        self.pre_generate = pre_generate
        self.num_threads = num_threads
        self.length = length
        
        assert(not with_symbols or custom_dataset_folder)
        self.with_symbols = with_symbols

        # Constants to match the Godot renderer
        self.VIEWPORT_WIDTH = 128
        self.VIEWPORT_HEIGHT = 32
        self.RIGHT_MARGIN = 5
        self.TOP_PADDING = 8
        self.LINE_WIDTH = 3

        self.digit_images = self.load_digit_images()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.pre_generate:
            start_time = time.time()
            self.data = []
            self.generate_data_single_threaded()
            end_time = time.time()
            print(f"Data generation time: {end_time - start_time:.2f} seconds")

    def trim_whitespace(self, image):
        """Trim excess whitespace around a digit image."""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Find non-zero rows and columns
        non_zero_rows = np.any(img_array > 0, axis=1)
        non_zero_cols = np.any(img_array > 0, axis=0)
        
        if not np.any(non_zero_rows) or not np.any(non_zero_cols):
            return image  # Image is empty
        
        # Get bounds
        top = np.where(non_zero_rows)[0][0]
        bottom = np.where(non_zero_rows)[0][-1]
        left = np.where(non_zero_cols)[0][0]
        right = np.where(non_zero_cols)[0][-1]
        
        # Add a small margin (1 pixel)
        top = max(0, top - 1)
        bottom = min(img_array.shape[0] - 1, bottom + 1)
        left = max(0, left - 1)
        right = min(img_array.shape[1] - 1, right + 1)
        
        # Crop the image
        return image.crop((left, top, right + 1, bottom + 1))

    def load_digit_images(self):
        # Define special symbols mapping
        digit_images = {str(i): [] for i in range(10)}
        if self.with_symbols:
            symbol_map = {
                '10': '+',
                '11': '-',
                '12': '<',
                '13': '>',
                '14': '='
            }

            # Initialize digit_images with symbols
            for sym_folder, sym_char in symbol_map.items():
                digit_images[sym_char] = []
        
        custom_counts = {str(i): 0 for i in range(10)}
        if self.with_symbols:
            for sym_char in symbol_map.values():
                custom_counts[sym_char] = 0

        # Load digits (0-9)
        for digit in range(10):
            folder_path = os.path.join(self.custom_dataset_folder, str(digit))
            if os.path.exists(folder_path):
                images = []
                for img_file in os.listdir(folder_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(folder_path, img_file)
                        image = Image.open(img_path).convert('L') 
                        if np.mean(image) > 127:
                            image = ImageOps.invert(image)
                        # Trim whitespace around the digit
                        image = self.trim_whitespace(image)
                        images.append(image)
                print(f"use {folder_path}, load {len(images)} images")
                digit_images[str(digit)].extend(images)
                custom_counts[str(digit)] = len(images)
        
        if self.with_symbols:
            # Load special symbols (+, -, <, >, =)
            for sym_folder, sym_char in symbol_map.items():
                folder_path = os.path.join(self.custom_dataset_folder, sym_folder)
                if os.path.exists(folder_path):
                    images = []
                    for img_file in os.listdir(folder_path):
                        if img_file.endswith('.png'):
                            img_path = os.path.join(folder_path, img_file)
                            image = Image.open(img_path).convert('L') 
                            if np.mean(image) > 127:
                                image = ImageOps.invert(image)
                            # Trim whitespace around the symbol
                            image = self.trim_whitespace(image)
                            images.append(image)
                    print(f"use {folder_path}, load {len(images)} images for symbol '{sym_char}'")
                    digit_images[sym_char].extend(images)
                    custom_counts[sym_char] = len(images)

        # Limit MNIST usage to complement up to 1000 digits per class (only for digits 0-9)
        mnist_digit_counts = {str(i): 0 for i in range(10)}
        mnist_digit_images = {str(i): [] for i in range(10)}
        
            
        for img, label in self.mnist_dataset:
            label_str = str(label)
            if mnist_digit_counts[label_str] < self.digits_per_class - custom_counts[label_str]:
                # Process the image
                processed_img = self._process_mnist_image(img)
                if processed_img is not None:
                    mnist_digit_images[label_str].append(processed_img)
                    mnist_digit_counts[label_str] += 1
            
            if all(count >= self.digits_per_class - custom_counts[str(i)] for i, count in mnist_digit_counts.items()):
                break
        for digit in range(10):
            digit_str = str(digit)
            digit_images[digit_str].extend(mnist_digit_images[digit_str])

        return digit_images
                
        
            
    def _process_mnist_image(self, img):
        """Helper function to process MNIST images of any format."""
        try:
            # Check type of img and convert appropriately
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                print(f"WARNING: Unexpected type for image: {type(img)}")
                return None
            
            # Make sure image is in correct mode
            if img.mode != 'L':
                img = img.convert('L')
                
            if np.mean(img) > 127:
                img = ImageOps.invert(img)
            
            # Trim whitespace around the digit
            return self.trim_whitespace(img)
        except Exception as e:
            print(f"Error processing MNIST image: {e}")
            return None



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.pre_generate:
            return self.data[idx]
        else:
            return self.generate_sample()

    def generate_sample(self):
        # 20% chance to generate a special symbol instead of a number
        if self.with_symbols and random.random() < 0.2:
            # Generate one special symbol
            symbol_choices = ['+', '-', '<', '>', '=']
            symbol = random.choice(symbol_choices)
            
            # Get the symbol image
            
            symbol_image = random.choice(self.digit_images[symbol])
            symbol_image = self.augment_digit(symbol_image, [1.1, 1.35])
            
            # Process a single symbol (using our Godot-like renderer)
            processed_image = self.render_godot_style([symbol_image])
            image_tensor = transforms.ToTensor()(processed_image)
            
            return image_tensor, symbol
        else:
            # Generate a number
            num_digits = random.randint(1, self.max_digits)
            if self.include_leading_zeros and random.random() < 0.1:
                # 10% chance to include leading zeros
                number_str = ''.join([str(random.randint(0, 9)) for _ in range(num_digits)])
                number_str = number_str.zfill(self.max_digits)
            else:
                number = random.randint(0, 10**num_digits - 1)
                number_str = str(number).zfill(num_digits)

            # Build the image by getting individual digit images
            digit_images = []
            for digit_char in number_str:
                # print(self.digit_images)
                digit_image = random.choice(self.digit_images[digit_char])
                digit_image = self.augment_digit(digit_image)
                digit_images.append(digit_image)

            # Use our Godot-like renderer
            processed_image = self.render_godot_style(digit_images)
            image_tensor = transforms.ToTensor()(processed_image)
            
            return image_tensor, number_str

    def augment_digit(self, image, custom_scale=None):
        # Random rotation
        rotation_degree = random.uniform(-10, 10)
        image = image.rotate(rotation_degree, fillcolor=0, resample=Image.BICUBIC)

        # Random zoom
        scale_factor_x = random.uniform(0.9, 1.2)
        scale_factor_y = random.uniform(0.9, 1.2)
        if custom_scale:
            scale_factor_x = random.uniform(custom_scale[0], custom_scale[1])
            scale_factor_y = random.uniform(custom_scale[0], custom_scale[1])
        new_size = (int(image.size[0]*scale_factor_x), int(image.size[1]*scale_factor_y))
        image = image.resize(new_size, Image.LANCZOS)

        # Random small shift (replaced ImageChops.offset with non-wrapping method)
        # max_dx = 2
        # max_dy = 2
        # dx = random.randint(-max_dx, max_dx)
        # dy = random.randint(-max_dy, max_dy)
        
        # Create a new image of the same size
        new_image = Image.new('L', image.size, color=0)
        
        # Paste the original image with an offset
        # Parts outside the boundaries will be clipped, not wrapped
        # new_image.paste(image, (dx, dy))
        new_image.paste(image)
        
        return new_image

    def render_godot_style(self, digit_images):
        """
        Render digit images in a style similar to the Godot renderer.
        This closely mimics the behavior of number_render_nn.gd set_curves method.
        Numbers will be stretched to fill the entire viewport height.
        """
        # Random vertical shifts for each digit
        vertical_shifts = [random.randint(-5, 5) for _ in digit_images]
        
        # Spacing modes with weighted distribution, focusing more on problematic cases
        spacing_mode_rand = random.random()
        if spacing_mode_rand < 0.35:
            spacing_mode = 'normal'
        elif spacing_mode_rand < 0.55:
            spacing_mode = 'tight'
        elif spacing_mode_rand < 0.75:
            spacing_mode = 'overlap'
        elif spacing_mode_rand < 0.85:
            spacing_mode = 'wide'
        else:
            spacing_mode = 'multi_overlap'
    
        # spacing_mode = 'multi_overlap'
        
        # Only use multi_overlap mode if we have 3 or more digits
        if spacing_mode == 'multi_overlap' and len(digit_images) < 3:
            spacing_mode = 'overlap'
        
        # For 3+ digits cases, increase the chance of difficult scenarios
        if len(digit_images) >= 3 and spacing_mode != 'multi_overlap' and random.random() < 0.1:
            spacing_mode = 'multi_overlap'
        
        # Set base spacing according to mode
        if spacing_mode == 'normal':
            base_spacing = random.randint(-1, 7)
        elif spacing_mode == 'tight':
            base_spacing = random.randint(-1, 2)
        elif spacing_mode == 'overlap':
            base_spacing = random.randint(-5, -2)
        elif spacing_mode == 'multi_overlap':
            base_spacing = random.randint(-8, -4)
        else:  # wide
            base_spacing = random.randint(6, 10)
        
        # Add some variation to the spacing
        if spacing_mode == 'multi_overlap':
            variation = 1
        elif spacing_mode == 'normal':
            variation = random.randint(3, 7)
        else:
            variation = random.randint(0, 3)
        
        spacings = [base_spacing + random.randint(-variation, variation) for _ in range(len(digit_images)-1)]

        # Calculate digit widths and limit overlap between adjacent digits
        digit_widths = [img.size[0] for img in digit_images]
        for i in range(len(digit_images)-1):
            # Calculate maximum allowed overlap based on 1/3 of the smaller width
            current_width = digit_widths[i]
            next_width = digit_widths[i+1]
            max_overlap = min(current_width, next_width) // 3
            spacings[i] = max(spacings[i], -max_overlap)
            
        
        # Calculate boundaries for placement
        max_digit_height = max(img.size[1] for img in digit_images)
        total_width = sum(img.size[0] for img in digit_images) + sum(spacings)
        
        # Create canvas with enough space for vertical shifts
        canvas_height = max_digit_height + 20  # Extra space for vertical shifts
        canvas = Image.new('L', (total_width, canvas_height), color=0)
        
        # Place the digits on the canvas
        x_offset = 0
        for i, img in enumerate(digit_images):
            y_offset = (canvas_height - img.size[1]) // 2 + vertical_shifts[i]
            
            temp_image = Image.new('L', canvas.size, color=0)
            temp_image.paste(img, (x_offset, y_offset))
            canvas = ImageChops.lighter(canvas, temp_image)
            
            if i < len(spacings):
                x_offset += img.size[0] + spacings[i]
            else:
                x_offset += img.size[0]
        
        # TRIM THE CANVAS: Remove empty rows from top and bottom
        canvas_array = np.array(canvas)
        non_zero_rows = np.any(canvas_array > 0, axis=1)
        
        if not np.any(non_zero_rows):
            # If image is empty, just return it as is
            top_row = 0
            bottom_row = canvas_height - 1
        else:
            # Find first and last non-empty rows
            top_row = np.where(non_zero_rows)[0][0]
            bottom_row = np.where(non_zero_rows)[0][-1]
            
            canvas_center = canvas_height // 2
            max_trim_top = canvas_center - canvas_height // 3
            max_trim_bottom = canvas_center + canvas_height // 3
            
            top_row = min(top_row, max_trim_top)
            bottom_row = max(bottom_row, max_trim_bottom)
        
        # Crop the canvas to remove empty rows, but not more than half of the pixels
        canvas = canvas.crop((0, top_row, canvas.width, bottom_row + 1))
        
        # Resize to fit viewport width while maintaining aspect ratio
        # Scale to fit full viewport height
        scale = self.VIEWPORT_HEIGHT / canvas.height
        new_width = int(canvas.width * scale)
        
        # Resize to match the scaling in Godot
        canvas = canvas.resize((new_width, self.VIEWPORT_HEIGHT), Image.LANCZOS)
        
        # Create the final image
        final_image = Image.new('L', (self.VIEWPORT_WIDTH, self.VIEWPORT_HEIGHT), color=0)
        
        # Right-align with 5px margin exactly as in Godot
        right_edge = self.VIEWPORT_WIDTH - self.RIGHT_MARGIN
        paste_x = right_edge - canvas.width
        paste_y = 0  # No vertical padding
        
        # Ensure we don't paste outside the bounds
        paste_x = max(0, paste_x)
        
        # Paste the content
        final_image.paste(canvas, (paste_x, paste_y))
        
        # Add some random noise to simulate rasterization artifacts (optional)
        if random.random() < 0.3:
            noise = np.random.randint(0, 10, size=(self.VIEWPORT_HEIGHT, self.VIEWPORT_WIDTH), dtype=np.uint8)
            noise_img = Image.fromarray(noise)
            final_image = ImageChops.lighter(final_image, noise_img)
        
        return final_image
    def visualize_sample(self, idx=None):
        """Visualize a single sample from the dataset."""
        if idx is None:
            image_tensor, label = self.generate_sample()
        else:
            image_tensor, label = self[idx]
        image = transforms.ToPILImage()(image_tensor)
        plt.figure(figsize=(6, 2))
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def visualize_batch(self, n=5, focused_on_digits=None):
        """
        Visualize a batch of samples from the dataset.
        
        Args:
            n: Number of samples to visualize
            focused_on_digits: If specified, generates samples with this many digits
        """
        plt.figure(figsize=(15, 3*n))
        
        for i in range(n):
            if focused_on_digits is not None:
                # Generate a sample with the specified number of digits
                while True:
                    image_tensor, label = self.generate_sample()
                    if isinstance(label, str) and len(label) == focused_on_digits:
                        break
            else:
                image_tensor, label = self.generate_sample()
                
            image = transforms.ToPILImage()(image_tensor)
            plt.subplot(n, 1, i+1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Label: {label}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def visualize_modes(self):
        """Visualize samples for each spacing mode to verify they look correct."""
        
        # Define all spacing modes
        modes = ['normal', 'tight', 'overlap', 'wide', 'multi_overlap']
        
        # Create a figure with subplots for each mode
        plt.figure(figsize=(15, 10))
        
        # For each mode, generate a 5-digit number and visualize
        for i, mode in enumerate(modes):
            # Store the original random state and modify render_godot_style temporarily
            original_render = self.render_godot_style
            
            # Create a modified render function that forces the specified mode
            def modified_render(digit_images, forced_mode=mode):
                # Set base spacing according to mode
                if forced_mode == 'normal':
                    base_spacing = random.randint(2, 6)
                elif forced_mode == 'tight':
                    base_spacing = random.randint(-1, 2)
                elif forced_mode == 'overlap':
                    base_spacing = random.randint(-5, -1)
                elif forced_mode == 'multi_overlap':
                    base_spacing = random.randint(-3, -2)
                else:  # wide
                    base_spacing = random.randint(6, 10)
                
                # Continue with the original implementation but with forced mode
                if forced_mode == 'multi_overlap':
                    variation = 1
                else:
                    variation = random.randint(0, 2)
                    
                spacings = [base_spacing + random.randint(-variation, variation) for _ in range(len(digit_images)-1)]
                
                if forced_mode == 'multi_overlap' and len(digit_images) >= 3:
                    for i in range(len(spacings)):
                        spacings[i] = -3 if i % 2 == 0 else -2
                
                # Rest of the original implementation...
                vertical_shifts = [random.randint(-3, 3) for _ in digit_images]
                max_digit_height = max(img.size[1] for img in digit_images)
                total_width = sum(img.size[0] for img in digit_images) + sum(spacings)
                canvas_height = max_digit_height + 8
                canvas = Image.new('L', (total_width, canvas_height), color=0)
                
                x_offset = 0
                for i, img in enumerate(digit_images):
                    y_offset = (canvas_height - img.size[1]) // 2 + vertical_shifts[i]
                    temp_image = Image.new('L', canvas.size, color=0)
                    temp_image.paste(img, (x_offset, y_offset))
                    canvas = ImageChops.lighter(canvas, temp_image)
                    
                    if i < len(spacings):
                        x_offset += img.size[0] + spacings[i]
                    else:
                        x_offset += img.size[0]
                
                max_content_height = self.VIEWPORT_HEIGHT - 16
                scale = max_content_height / canvas.size[1]
                
                new_width = int(canvas.size[0] * scale)
                new_height = int(canvas.size[1] * scale)
                canvas = canvas.resize((new_width, new_height), Image.LANCZOS)
                
                final_image = Image.new('L', (self.VIEWPORT_WIDTH, self.VIEWPORT_HEIGHT), color=0)
                
                right_edge = self.VIEWPORT_WIDTH - self.RIGHT_MARGIN
                paste_x = right_edge - canvas.size[0]
                paste_y = self.TOP_PADDING
                
                paste_x = max(0, paste_x)
                final_image.paste(canvas, (paste_x, paste_y))
                
                return final_image
            
            # Replace the render function temporarily
            self.render_godot_style = modified_render
            
            # Create a subplot
            plt.subplot(len(modes), 1, i+1)
            
            # Generate a 5-digit sample
            while True:
                image_tensor, label = self.generate_sample()
                if isinstance(label, str) and len(label) == 5:
                    break
            
            # Display the image
            image = transforms.ToPILImage()(image_tensor)
            plt.imshow(image, cmap='gray')
            plt.title(f"Mode: {mode} - Label: {label}")
            plt.axis('off')
            
            # Restore the original render function
            self.render_godot_style = original_render
        
        plt.tight_layout()
        plt.show()

    def generate_data_single_threaded(self):
        data_length = self.__len__()
        self.data = []

        for i in range(data_length):
            self.data.append(self.generate_sample())
            # if i % 1000 == 0:
            #     print(f"Generated {i}/{data_length} samples")

        random.shuffle(self.data)


# To maintain compatibility with original generator
class HandwrittenNumbersDataset(ImprovedHandwrittenNumbersDataset):
    pass

