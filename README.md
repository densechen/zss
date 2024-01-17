# Zero-Shot Video Sampling from Image

## Comparison with baseline

| Method   |                 A cat is running on the grass.                 |             An astronaut is skiing down the hill.              |                 A horse galloping on a street.                 |
| :------- | :------------------------------------------------------------: | :------------------------------------------------------------: | :------------------------------------------------------------: |
| T2V-Zero | <img src="video/comparison_with_baseline/a1.gif" width="256">  | <img src="video/comparison_with_baseline/a2.gif" width="256" > | <img src="video/comparison_with_baseline/a3.gif" width="256" > |
| Ours     | <img src="video/comparison_with_baseline/b1.gif" width="256" > | <img src="video/comparison_with_baseline/b2.gif" width="256" > | <img src="video/comparison_with_baseline/b3.gif" width="256" > |

## Comparison with different diffusion models

| Model  | Prompt                                                            |                                     Sampled                                      |
| :----: | :---------------------------------------------------------------- | :------------------------------------------------------------------------------: |
| SDv1.4 | A man is riding a bicycle in the sunshine.                        | <img src="video/comparison_with_different_diffusion_models/a1.gif" width="256" > |
| SDv1.5 | A cat is wearing sunglasses and working as a lifeguard at a pool. | <img src="video/comparison_with_different_diffusion_models/a2.gif" width="256" > |
| DPv1.0 | A cute cat running in a beautiful meadow.                         | <img src="video/comparison_with_different_diffusion_models/a3.gif" width="256" > |
| DPv2.0 | A group of squirrels rowing crew.                                 | <img src="video/comparison_with_different_diffusion_models/a4.gif" width="256" > |
|   OJ   | A beautiful girl.                                                 | <img src="video/comparison_with_different_diffusion_models/a5.gif" width="256" > |