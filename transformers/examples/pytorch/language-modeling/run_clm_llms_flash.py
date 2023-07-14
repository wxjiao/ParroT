# Need to call this before importing transformers.
from flash_attention.bloom_flash_attention import (
    replace_bloom_attn_with_flash_attn,
)

replace_bloom_attn_with_flash_attn()

from run_clm_llms import main

if __name__ == "__main__":
    main()
