#!/bin/bash
micromamba run -n personaplex python -m moshi.server --quantize 4bit --port 8998
