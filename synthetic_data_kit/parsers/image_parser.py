"""Image parser utilities."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Dict, List


class ImageParser:
    """Parser that converts image files to Base64 strings."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}

    def parse(self, file_path: str) -> List[Dict[str, str]]:
        """Read an image from ``file_path`` and return Base64 encoded content.

        Args:
            file_path: Path to the image on disk.

        Returns:
            List containing a single dictionary with the Base64 encoded image
            string under the ``image`` key and an empty ``text`` value for
            interface parity with other parsers.

        Raises:
            FileNotFoundError: If ``file_path`` does not point to a file.
            ValueError: If the file extension is not a supported image format.
        """

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{path.suffix}'. "
                f"Supported extensions: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        image_bytes = path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return [{"text": "", "image": image_base64}]

    def save(self, image_base64: str, output_path: str) -> None:
        """Save a Base64 encoded image to disk.

        Args:
            image_base64: The Base64 encoded image string.
            output_path: Path where the decoded image should be written.
        """

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        image_bytes = base64.b64decode(image_base64)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
