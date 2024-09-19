import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from numbers_generator import HandwrittenNumbersDataset
import random
import torch

# Настройка воспроизводимости
seed = 46
random.seed(seed)
torch.manual_seed(seed)

# Загрузка MNIST датасета
mnist_dataset = MNIST(root='./data', train=True, download=True)

# Создание экземпляра HandwrittenNumbersDataset
dataset = HandwrittenNumbersDataset(
    custom_dataset_folder='/Users/sergejsorin/work/math/lib/mnist_improve/local_data/sorted',  # Замените на путь к вашему кастомному датасету
    mnist_dataset=mnist_dataset,
    max_digits=5,
    length=1000,  # Уменьшаем длину для быстрой генерации
    include_leading_zeros=True,
    seed=seed,
    pre_generate=False  # Устанавливаем False для генерации на лету
)

# Создание фигуры с 6 подграфиками
fig, axs = plt.subplots(4, 3, figsize=(12, 8))
# fig.suptitle("Примеры сгенерированных изображений чисел", fontsize=16)

# Генерация и отображение 6 случайных изображений
for i, ax in enumerate(axs.flatten()):
    image_tensor, label = dataset.generate_sample()
    image = image_tensor.squeeze().numpy()  # Преобразуем тензор в numpy массив
    
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Число: {label}", fontsize=16)
    ax.axis('off')

plt.tight_layout()
plt.savefig('images/generated_samples.png', dpi=300, bbox_inches='tight')
plt.show()