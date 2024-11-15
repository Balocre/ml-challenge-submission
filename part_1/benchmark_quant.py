import torch
import torch.utils.benchmark as benchmark
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import build_imagenet1000_classes, calculate_model_size, quantize_model

torch.manual_seed(42)

MODEL_ID = "dinov2_vitb14_reg_lc"
MODEL = torch.hub.load("facebookresearch/dinov2", MODEL_ID)
BENCHMARK = {
    "normal": MODEL,
    "quantized - qint8": quantize_model(MODEL, torch.qint8),
    "quantized - float16": quantize_model(MODEL, torch.float16),
}
IM1000_CLASSES = build_imagenet1000_classes("imagenet-1000_words.txt")


def get_model_accuracy_topk(
    model, dataset_path="dataset/", k=5, n_batch=None, batch_size=8
):
    """A function that calculates the top-k accuracy of a model over an
    ImageFolder dataset

    :param model: The model
    :param dataset_path: Path to the dataset root
    :param k: The number of items to include in the top
    :param n_batch: A paramater to only iterate of the N first batches
    :param batch_size: The bacth size

    :return: The accuracy of the model as an integer
    """

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.Resize((70, 70)),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    tp = 0

    if n_batch is None:
        n_batch = len(loader)

    dataiter = iter(loader)
    torch.manual_seed(42)  # set seed to guarantee same oreder in case of partial eval
    for _ in tqdm(range(0, n_batch), leave=False):
        x_batch, y_batch = next(dataiter)
        y_hat = model(x_batch)

        # tiny-imagenet-200 is a subset of imagenet-1000, plus because of the way
        # ImageFolder is built, class numbers need to be remapped, so we get the
        # corresponding imagenet-1000 class number of labels
        y_im1000_batch = torch.tensor(
            [IM1000_CLASSES.index(dataset.classes[i]) for i in y_batch]
        )

        top_k_batch = torch.topk(y_hat, k, dim=1).indices

        t = torch.hstack([top_k_batch, y_im1000_batch[:, None]])

        for row in t:
            _, counts = torch.unique(row, return_counts=True)
            # the only case where there are 2 identical elements is if ground-truth is
            # in top-k
            if torch.any(counts == 2):
                tp += 1

    return tp / len(dataset)


def benchmark_accuracy(topk=5, n_batch=None):
    print("\n===== Running ACCURACY benchmark ======\n")
    print(f"Evaluating models accuracy using TOP-{topk}")

    for name, model in BENCHMARK.items():
        model_accuracy = get_model_accuracy_topk(model, k=5, n_batch=n_batch)
        print(f"Accuracy ({name}):\t{model_accuracy}")


def benchmark_inference_time_torch(input_W_H=518):
    """An inference time benchmark for torch models"""

    print("\n===== Running INFERENCE benchmark ======\n")
    num_threads = torch.get_num_threads()

    results = list()
    # try varying batch sizes use default DINOv2 sizes
    for batch_size in tqdm([1, 2, 4, 8], leave=False):
        x = torch.randn(batch_size, 3, input_W_H, input_W_H)

        for name in BENCHMARK.keys():
            t = benchmark.Timer(
                stmt=f'BENCHMARK["{name}"](x)',
                setup="from __main__ import BENCHMARK",
                globals={"x": x},
                num_threads=num_threads,
                label="ViT Forward",
                sub_label=f"{x.shape}",
                description=f"{name}",
            )
            m = t.blocked_autorange(min_run_time=1)
            results.append(m)

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_size():
    print("\n===== Running SIZE benchmark ======\n")

    print("Estimated size in memory")
    for name, model in BENCHMARK.items():
        model_size_b, model_size_mb = calculate_model_size(model)
        print(
            f"Size ({name}):\t disk - {model_size_b} MB  | memory - {model_size_mb} MB "
        )


if __name__ == "__main__":
    benchmark_accuracy()
    benchmark_inference_time_torch()
    benchmark_size()
