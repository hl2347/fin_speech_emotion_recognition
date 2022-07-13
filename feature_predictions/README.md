Once the FINs are trained, they are chained together horizontally to generate predicted features.
In order to chain FINs of different topologies, keras functional API was used(instead of the sequential module), and the weights are applied into the frames generated in the functional API.

Since the second(Crema-D) dataset is too big, the feature generation process was split into parts.