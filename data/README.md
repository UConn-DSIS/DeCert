# HelpSteer3 Data Source

DeCert uses samples from the public
[`preference/train`](https://huggingface.co/datasets/nvidia/HelpSteer3/viewer/preference/train)
split of NVIDIA HelpSteer3.

The pipeline expects a local JSONL file containing one object per line. Each
object must contain the original `context`, `domain`, `language`, `response1`,
`response2`, `overall_preference`, and `individual_preference` fields. Pass
its path with the `DATA` environment variable or the relevant `--data` or
`--input` command-line option.

HelpSteer3 follows the upstream
[CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/). The code
in the rest of this repository follows the repository's MIT license.
