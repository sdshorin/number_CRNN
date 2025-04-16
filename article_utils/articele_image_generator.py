import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from numbers_generator import HandwrittenNumbersDataset

from new_generator import ImprovedHandwrittenNumbersDataset
import random
import torch


from config import get_custom_dataset_folder

def main():
    # seed = 49
    # random.seed(seed)
    # torch.manual_seed(seed)
    seed = random.randint(0, 9999999)


    mnist_dataset = MNIST(root='./data', train=True, download=True)


    # dataset = HandwrittenNumbersDataset(
    dataset = ImprovedHandwrittenNumbersDataset(
        custom_dataset_folder=get_custom_dataset_folder(),
        mnist_dataset=mnist_dataset,
        max_digits=5,
        length=1000,
        include_leading_zeros=True,
        seed=seed,
        pre_generate=False
    )


    fig, axs = plt.subplots(4, 3, figsize=(12, 8))



    for i, ax in enumerate(axs.flatten()):
        image_tensor, label = dataset.generate_sample()
        image = image_tensor.squeeze().numpy()
        
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Число: {label}", fontsize=16)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('images/generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
