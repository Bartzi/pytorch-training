from torchvision import datasets


class MNIST(datasets.MNIST):

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return {
            "images": image,
            "labels": label
        }
