## FIN Results

### Only Dense

|   | R2 | MSE | # params |
| --- | --- | --- | --- |
| ZCR | 0.95 | 0.001 | 2.3M |
| RMS | 0.99 | 6.7e-06 | 3.2M |
| Chroma | 0.90 | 0.60 | 3.8M |
| Melspectrogram | 0.18 | 38914.78 | 4.2M |
| MFCC | \-0.12 |  | 2.9M |

*MFCC FIN trained to predict the normalized feature showed an R-squared of 0.61

---


### With Convolution

|   | R2 | MSE | # params |
| --- | --- | --- | --- |
| ZCR | 0.74 | 0.006 | 8.2K |
| RMS |  |  |  |
| Chroma | 0.83 | 0.01 | 149K |
| Melspectrogram | 0.38 | 39.67 | 148K |
| MFCC | \-0.19 | 12107.74 | 265K |

