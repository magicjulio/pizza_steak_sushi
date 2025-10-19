import io
from typing import List, Union

import torch
from PIL import Image
from werkzeug.datastructures import FileStorage

# Optional: enable HEIC/HEIF support (common on iPhone photos)
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:
    # If not installed, JPEG/PNG will still work; HEIC may fail without this package
    pass


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: Union[str, io.BytesIO, FileStorage],
    class_names: List[str] = None,
    transform=None,
):
    """Make a prediction on an image (from path or uploaded file).

    Args:
        model: Trained PyTorch image classification model.
        image_path: File path or file-like object (e.g., Flask FileStorage).
        class_names: Optional list of class names.
        transform: Optional torchvision transform to apply (expects PIL Image -> Tensor pipeline).

    Returns:
        (pred_index: int, pred_probs: torch.Tensor [1, num_classes])
    """

    # Load image from path or file-like
    if hasattr(image_path, "read"):
        # It's a file-like (e.g., Flask's FileStorage)
        stream = image_path.stream if hasattr(image_path, "stream") else image_path
        try:
            # Ensure stream is at beginning for PIL
            if hasattr(stream, "seek"):
                stream.seek(0)
        except Exception:
            pass
        image = Image.open(stream).convert("RGB")
    else:
        image = Image.open(str(image_path)).convert("RGB")

    # Apply transforms
    if transform is not None:
        img_tensor = transform(image)
    else:
        # Fallback minimal transform similar to ToTensor
        from torchvision import transforms as T

        img_tensor = T.ToTensor()(image)

    # Model inference
    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor.unsqueeze(0))  # [1, C]

    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())

    return pred_idx, probs



