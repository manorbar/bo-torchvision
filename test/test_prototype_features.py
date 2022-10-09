import pytest
import torch
from torchvision.prototype import features


def test_isinstance():
    assert isinstance(
        features.Label([0, 1, 0], categories=["foo", "bar"]),
        torch.Tensor,
    )


def test_wrapping_no_copy():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    assert label.data_ptr() == tensor.data_ptr()


def test_to_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    label_to = label.to(torch.int32)

    assert type(label_to) is features.Label
    assert label_to.dtype is torch.int32
    assert label_to.categories is label.categories


def test_to_feature_reference():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"]).to(torch.int32)

    tensor_to = tensor.to(label)

    assert type(tensor_to) is torch.Tensor
    assert tensor_to.dtype is torch.int32


def test_clone_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    label_clone = label.clone()

    assert type(label_clone) is features.Label
    assert label_clone.data_ptr() != label.data_ptr()
    assert label_clone.categories is label.categories


def test_requires_grad__wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.float32)
    label = features.Label(tensor, categories=["foo", "bar"])

    assert not label.requires_grad

    label_requires_grad = label.requires_grad_(True)

    assert type(label_requires_grad) is features.Label
    assert label.requires_grad
    assert label_requires_grad.requires_grad


def test_other_op_no_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    # any operation besides .to() and .clone() will do here
    output = label * 2

    assert type(output) is torch.Tensor


@pytest.mark.parametrize(
    "op",
    [
        lambda t: t.numpy(),
        lambda t: t.tolist(),
        lambda t: t.max(dim=-1),
    ],
)
def test_no_tensor_output_op_no_wrapping(op):
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    output = op(label)

    assert type(output) is not features.Label


def test_inplace_op_no_wrapping():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    output = label.add_(0)

    assert type(output) is torch.Tensor
    assert type(label) is features.Label


def test_wrap_like():
    tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
    label = features.Label(tensor, categories=["foo", "bar"])

    # any operation besides .to() and .clone() will do here
    output = label * 2

    label_new = features.Label.wrap_like(label, output)

    assert type(label_new) is features.Label
    assert label_new.data_ptr() == output.data_ptr()
    assert label_new.categories is label.categories
