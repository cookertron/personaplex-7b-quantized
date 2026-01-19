#!/bin/bash
export HF_TOKEN=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
micromamba run -n personaplex python -m moshi.server --quantize 4bit --port 8998
