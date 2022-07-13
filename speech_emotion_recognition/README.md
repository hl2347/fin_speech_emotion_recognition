## Baseline Model & Consolidated Model

The consolidated model classifies audio files in frame-level. Although this approach allows the FIN weights to be trained because the FINs are part of the consolidated model, it deviates from the baseline model in that this model outputs one emotion for one frame while the baseline model outputs one emotion for one audio file.
### Baseline Model
Accuracy :  61.61154508590698 %
| precision | recall | f1-score | support |
| --- | --- | --- | --- |
| angry | 0.81 | 0.65 | 0.72 | 894 |
| calm | 0.69 | 0.82 | 0.75 | 80 |
| disgust | 0.54 | 0.53 | 0.54 | 855 |
| fear | 0.62 | 0.55 | 0.58 | 883 |
| happy | 0.54 | 0.61 | 0.58 | 855 |
| neutral | 0.56 | 0.57 | 0.57 | 753 |
| sad | 0.57 | 0.67 | 0.62 | 833 |
| surprise | 0.86 | 0.84 | 0.85 | 320 |
| accuracy |  | 0.62 | 5473 |
| macro avg | 0.65 | 0.66 | 0.65 | 5473 |
| weighted avg | 0.63 | 0.62 | 0.62 | 5473 |
---
### Consolidated Model
Untrainable: 30.9%
Trainable: 29%

### Baseline Compare Models
