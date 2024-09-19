# numbers_generator.py
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

class HandwrittenNumbersDataset(Dataset):
    digits_per_class = 1000

    def __init__(self, custom_dataset_folder, mnist_dataset, max_digits=5, length=100_000, include_leading_zeros=False, seed=None, pre_generate=False, num_threads=1):
        self.custom_dataset_folder = custom_dataset_folder
        self.mnist_dataset = mnist_dataset
        self.max_digits = max_digits
        self.include_leading_zeros = include_leading_zeros
        self.seed = seed
        self.pre_generate = pre_generate
        self.num_threads = num_threads
        self.length = length

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

    def load_digit_images(self):

        digit_images = {str(i): [] for i in range(10)}
        custom_counts = {}

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
                        images.append(image)
                print(f"use {folder_path}, load {len(images)} images")
                digit_images[str(digit)].extend(images)
                custom_counts[str(digit)] = len(images)
            else:
                custom_counts[str(digit)] = 0

        # Limit MNIST usage to complement up to 1000 digits per class
        mnist_digit_counts = {str(i): 0 for i in range(10)}
        mnist_digit_images = {str(i): [] for i in range(10)}
        for img, label in self.mnist_dataset:
            label_str = str(label)
            if mnist_digit_counts[label_str] < self.digits_per_class - custom_counts[label_str]:
                if np.mean(img) > 127:
                    img = ImageOps.invert(img)
                mnist_digit_images[label_str].append(img)
                mnist_digit_counts[label_str] += 1
            if all(count >= self.digits_per_class - custom_counts[str(i)] for i, count in mnist_digit_counts.items()):
                break

        # Combine custom and MNIST images
        for digit in digit_images.keys():
            digit_images[digit].extend(mnist_digit_images[digit])

        return digit_images

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.pre_generate:
            return self.data[idx]
        else:
            return self.generate_sample()

    def generate_sample(self):
        num_digits = random.randint(1, self.max_digits)
        if self.include_leading_zeros and random.random() < 0.1:
            # 10% chance to include leading zeros
            number_str = ''.join([str(random.randint(0, 9)) for _ in range(num_digits)])
            number_str = number_str.zfill(self.max_digits)
        else:
            number = random.randint(0, 10**num_digits - 1)
            number_str = str(number).zfill(num_digits)

        # Build the image by concatenating digit images with random spacing and vertical position
        digit_images = []
        for digit_char in number_str:
            digit_image = random.choice(self.digit_images[digit_char])
            digit_image = self.augment_digit(digit_image)
            digit_images.append(digit_image)

        composite_image = self.concatenate_digits(digit_images)
        processed_image = self.process_image(composite_image)

        image_tensor = transforms.ToTensor()(processed_image)
        label = number_str

        return image_tensor, label

    def augment_digit(self, image):
        transform_list = []

        # Random rotation
        rotation_degree = random.uniform(-10, 10)
        image = image.rotate(rotation_degree, fillcolor=0)

        # Random zoom
        scale_factor = random.uniform(0.9, 1.1)
        new_size = (int(image.size[0]*scale_factor), int(image.size[1]*scale_factor))
        image = image.resize(new_size, Image.LANCZOS)

        # Random shift
        max_dx = 4
        max_dy = 4
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        image = ImageChops.offset(image, dx, dy)

        return image

    def concatenate_digits(self, digit_images):
        # Random vertical shifts and spacing
        max_digit_height = max(img.size[1] for img in digit_images)
        vertical_shifts = [random.randint(-5, 5) for _ in digit_images]
        spacings = [random.randint(-12, 10) for _ in range(len(digit_images)-1)]

        # Calculate total width
        total_width = sum(img.size[0] for img in digit_images) + sum(spacings)
        canvas_height = max_digit_height + 10  # Extra space for vertical shifts

        new_image = Image.new('L', (total_width, canvas_height), color=0)

        x_offset = 0
        for i, im in enumerate(digit_images):
            y_offset = (canvas_height - im.size[1]) // 2 + vertical_shifts[i]
            temp_image = Image.new('L', new_image.size, color=0)
            temp_image.paste(im, (x_offset, y_offset))
            new_image = ImageChops.lighter(new_image, temp_image)

            if i < len(spacings):
                x_offset += im.size[0] + spacings[i]
            else:
                x_offset += im.size[0]

        return new_image

    def process_image(self, image):
        # Scale image so that the maximum height of the content is 30 pixels
        max_content_height = 30
        w_percent = (max_content_height / float(image.size[1]))
        new_width = int((float(image.size[0]) * float(w_percent)))
        image = image.resize((new_width, max_content_height), Image.LANCZOS)

        # Add 1-pixel margins to make height 32 pixels
        image = ImageOps.expand(image, border=(0, 1), fill=0)  # Black background

        # Right-align and crop or pad if necessary
        if image.size[0] > 128:
            image = image.crop((image.size[0] - 128, 0, image.size[0], image.size[1]))
        else:
            # Add padding to the left
            padding = 128 - image.size[0]
            image = ImageOps.expand(image, border=(padding, 0, 0, 0), fill=0)  # Black background

        return image

    def visualize_sample(self, idx=None):
        if idx is None:
            image_tensor, label = self.generate_sample()
        else:
            image_tensor, label = self[idx]
        image = transforms.ToPILImage()(image_tensor)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

    def generate_data_single_threaded(self):
        data_length = self.__len__()
        self.data = []

        for i in range(data_length):
            self.data.append(self.generate_sample())

        random.shuffle(self.data)


from torchvision.datasets import MNIST



def test_dataset():
    mnist_dataset = MNIST(root='./data', train=True, download=True)

    length=100_000
    seed = random.randint(0, 9999999)
    # seed = 2122224
    print(f"Seed: {seed}")

    dataset = HandwrittenNumbersDataset(
        custom_dataset_folder=get_custom_dataset_folder(),
        mnist_dataset=mnist_dataset,
        max_digits=5,
        length=length,
        include_leading_zeros=True,
        seed=seed,
        pre_generate=True
    )

    dataset.generate_sample()    
    import os
    import hashlib
    from PIL import Image
    import io

    def image_to_hash(image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

    def check_and_save_images(dataset, num_images=100000, save_interval=1000):
        os.makedirs('raw_images', exist_ok=True)
        os.makedirs('raw_duplicates', exist_ok=True)
        image_hashes = {}
        duplicate_count = 0

        for i in range(num_images):
            image_tensor, label = dataset[i]
            image = transforms.ToPILImage()(image_tensor)
            
            image_hash = image_to_hash(image)
            number = ''.join(map(str, label.numpy()))
            
            if image_hash in image_hashes:
                duplicate_count += 1

                dup_filename = f"raw_duplicates/{image_hash}_{number}_dup{duplicate_count}.png"
                image.save(dup_filename)

                orig_filename = f"raw_duplicates/{image_hash}_{number}_orig.png"
                orig_image, _ = dataset[image_hashes[image_hash]]
                orig_image = transforms.ToPILImage()(orig_image)
                orig_image.save(orig_filename)
            else:
                image_hashes[image_hash] = i
                if i % save_interval == 0:
                    filename = f"raw_images/{image_hash}_{number}.png"
                    image.save(filename)
            
            if i % 1000 == 0:
                print(f"Processed {i} images. Duplicates so far: {duplicate_count}")

        print(f"Total images processed: {num_images}")
        print(f"Total unique images: {len(image_hashes)}")
        print(f"Total duplicates: {duplicate_count}")

    # Run the check
    check_and_save_images(dataset, length)

if __name__ == "__main__":
    test_dataset()

