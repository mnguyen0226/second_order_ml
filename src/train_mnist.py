import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from network import ConvMNIST
from ada_hessian import AdaHessian
import time

# source: https://github.com/pytorch/examples/blob/master/mnist/main.py

# optimizer_type = "AdaHessian"
optimizer_type = "SGD"
# optimizer_type = "Adam"
# optimizer_type = "Adagrad"
# optimizer_type = "AdamW"
# optimizer_type = "RMSProp"


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Trains model with different hyperparameters on training MNIST dataset
    Arguments:
        args: place-holder for arguments parsers in main()
        model: ConvMNIST
        device: cpu or gpu
        train_loader: training MNIST data processed by DataLoader()
        optimizer: first or second order optimizer
        epoch: number of epoch for training
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if optimizer_type == "AdaHessian":
            loss.backward(create_graph=True)
        else:
            loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tIteration cost: {:.3f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    time.time() - start,
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    """
    Tests trained model on testing MNIST dataset
    Arguments:
        model: ConvMNIST
        device: cpu or gpu
        test_loader: testing MNIST data processed by DataLoader()
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ConvMNIST().to(device)

    # Specify the optimizer
    if optimizer_type == "AdaHessian":
        optimizer = AdaHessian(model.parameters())
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.1)
    elif optimizer_type == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=0.1)
    elif optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    elif optimizer_type == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        raise TypeError(f"invalid optimizer type: {optimizer_type}")

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "../model/mnist_cnn.pt")


if __name__ == "__main__":
    main()
