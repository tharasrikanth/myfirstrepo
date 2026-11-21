import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image
from openvino import opset13 as ops


def positive_int(value):
    """Parse a positive integer for image dimensions and iteration counts."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an image-processing workload on the Intel NPU."
    )
    parser.add_argument("--height", type=positive_int, default=512)
    parser.add_argument("--width", type=positive_int, default=512)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().with_name("Dog_Image.png"),
        help="source image to process",
    )
    parser.add_argument(
        "--channels",
        type=positive_int,
        default=16,
        help="number of feature channels in the internal convolution layers",
    )
    parser.add_argument("--repeats", type=positive_int, default=10)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("NPU_output.png"),
        help="PNG path for the final processed image",
    )
    args = parser.parse_args()
    print(args)
    return args


def save_png(output, output_path):
    """Convert the model's FP16 RGB output into a viewable 8-bit PNG."""
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
        raise ValueError(f"Expected one RGB output image, got shape {output.shape}")

    # Convert NCHW to RGB layout and stretch each color channel independently.
    pixels = output[0].transpose(1, 2, 0).astype(np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError("Model output contains NaN or infinite values")

    low, high = np.percentile(
        pixels, [1, 99], axis=(0, 1), keepdims=True
    )
    spread = np.where(high > low, high - low, 1)
    pixels_8bit = np.rint(
        np.clip((pixels - low) / spread, 0, 1) * 255
    ).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels_8bit).save(output_path, format="PNG")


def load_input_image(input_path, height, width):
    """Load an RGB image, fit it inside the model dimensions, and make FP16 NCHW data."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    with Image.open(input_path) as source:
        source = source.convert("RGB")
        resized = source.copy()
        resized.thumbnail((width, height), Image.Resampling.LANCZOS)

    # Center the resized image on a black canvas without stretching or cropping it.
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    offset = ((width - resized.width) // 2, (height - resized.height) // 2)
    canvas.paste(resized, offset)

    # Convert RGB pixels from [0, 255] to FP16 values in [0, 1], then use NCHW.
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    return pixels.transpose(2, 0, 1)[None, ...].astype(np.float16)


def build_model(height, width, channels):
    # Use a fixed seed so the randomly generated filter weights are repeatable.
    rng = np.random.default_rng(42)
    image = ops.parameter(
        [1, 3, height, width], dtype=ov.Type.f16, name="image"
    )

    # Five 3x3 convolutions: RGB to feature maps, three feature layers,
    # then feature maps back to RGB. Padding keeps the image size unchanged.
    layer_channels = [channels, channels, channels, channels, 3]
    input_channels = 3
    result = image
    convolution_shapes = []

    for layer_index, output_channels in enumerate(layer_channels):
        # OpenVINO convolution filters use [output, input, height, width].
        filter_shape = (output_channels, input_channels, 3, 3)
        filters = rng.normal(0, 0.02, size=filter_shape).astype(np.float16)
        convolution_shapes.append((input_channels, output_channels))

        result = ops.convolution(
            result,
            ops.constant(filters),
            strides=[1, 1],
            pads_begin=[1, 1],
            pads_end=[1, 1],
            dilations=[1, 1],
        )

        # Apply an activation after feature layers, not after the RGB output.
        if layer_index < len(layer_channels) - 1:
            result = ops.relu(result)

        input_channels = output_channels

    # Keep the source image visible while the convolution stack adds its effect.
    result = ops.add(image, result)
    model = ov.Model([result], [image], "image_processing_simulation")
    return model, convolution_shapes


def main():
    args = parse_args()
    core = ov.Core()
    print("OpenVINO devices:", core.available_devices)

    # Never silently fall back to CPU or GPU: this test is specifically for NPU.
    if "NPU" not in core.available_devices:
        raise RuntimeError(
            "OpenVINO did not find an NPU. Check that the Intel NPU driver is installed."
        )

    image = load_input_image(args.input, args.height, args.width)
    print(f"Source image: {args.input.resolve()}")
    model, convolution_shapes = build_model(
        args.height, args.width, args.channels
    )
    print("Compiling the image-processing model for NPU...")
    compiled_model = core.compile_model(model, "NPU")
    print(
        "Compiled execution device:",
        compiled_model.get_property(ov.properties.execution_devices),
    )
    request = compiled_model.create_infer_request()

    # Estimate convolution work: output pixels times filter and channel products.
    multiply_accumulates = sum(
        args.height * args.width * 3 * 3 * in_channels * out_channels
        for in_channels, out_channels in convolution_shapes
    )
    print(
        f"Input: 1 x 3 x {args.height} x {args.width} "
        f"({image.nbytes / (1024 ** 2):.1f} MiB, FP16)"
    )
    print(f"Convolution layers: {len(convolution_shapes)}")
    print(f"Estimated work: {multiply_accumulates / 1e9:.2f} billion MACs per inference")

    # Warm up the compiled graph; warm-up runs are not included in timing.
    for _ in range(args.warmup):
        result = request.infer({0: image})

    start = time.perf_counter()
    for _ in range(args.repeats):
        result = request.infer({0: image})
    elapsed = time.perf_counter() - start

    output = np.asarray(result[compiled_model.output(0)])
    average_seconds = elapsed / args.repeats
    print(f"Completed {args.repeats} timed NPU inferences.")
    print(f"Average inference time: {average_seconds * 1000:.2f} ms")
    print(f"Throughput: {1 / average_seconds:.2f} images/second")
    print("Output shape:", output.shape)
    print("Output sample:", output[0, :, 0, :5])
    save_png(output, args.output)
    print("Saved processed image:", args.output.resolve())
    # Open the saved PNG in the computer's default image viewer.
    with Image.open(args.output) as saved_image:
        saved_image.show(title="NPU processed image")


if __name__ == "__main__":
    main()
