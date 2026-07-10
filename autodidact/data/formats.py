"""Binary formats shared by writers, readers, and integrity checks."""

import numpy as np

TOKEN_DTYPE = np.dtype("<u2")
INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("token_count", "<u4"),
        ("utf8_bytes", "<u4"),
        ("content_sha256", "S32"),
    ]
)
