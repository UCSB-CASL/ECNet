# ECNet
### Error-correcting neural network for semi-Lagrangian advection in the level-set method

#### Abstract

We present a machine learning framework that blends image super-resolution technologies with scalar transport in the level-set method.  Here, we investigate whether we can compute on-the-fly data-driven corrections to minimize numerical viscosity in the coarse-mesh evolution of an interface.  The proposed system's starting point is the semi-Lagrangian formulation.  And, to reduce numerical dissipation, we introduce an error-quantifying multilayer perceptron.  The role of this neural network is to improve the numerically estimated surface trajectory.  To do so, it processes localized level-set, velocity, and positional data in a single time frame for select vertices near the moving front.  Our main contribution is thus a novel machine-learning-augmented transport algorithm that operates alongside selective redistancing and alternates with conventional advection to keep the adjusted interface trajectory smooth.  Consequently, our procedure is more efficient than full-scan convolutional-based applications because it concentrates computational effort only around the free boundary.  Also, we show through various tests that our strategy is effective at counteracting both numerical diffusion and mass loss.  In passive advection problems, for example, our method can achieve the same precision as the baseline scheme at twice the resolution but at a fraction of the cost.  Similarly, our hybrid technique can produce feasible solidification fronts for crystallization processes.  On the other hand, highly deforming or lengthy simulations can precipitate bias artifacts and inference deterioration.  Likewise, stringent design velocity constraints can impose certain limitations, especially for problems involving rapid interface changes.  In the latter cases, we have identified several opportunity avenues to enhance robustness without forgoing our approach's basic concept.  Despite these circumstances, we believe all the above assets make our framework attractive to parallel level-set algorithms.  Its appeal resides in the possibility of avoiding further mesh refinement and decreasing expensive communications between computing nodes.

#### Contents

We include the following files:
1. `requirements.txt`: Contains `python` virtual environment packages to load the neural network in TensorFlow/Keras.
2. `mass_nnet.h5`: Tensorflow/Keras model with no optimizer.
3. `mass_nnet.json`: JSON version of neural network with hidden-layer weights encoded in `Base64` but represented as `ASCII` text.  It does not include the `h`-denormalization operation that comes at the end of `mass_nnet.h5`.
4. `mass_pca_scaler.pkl`: PCA scaler stored in `pickle` format.
5. `mass_pca_scaler.json`: JSON version of PCA scaler with plain-valued parameters.
6. `mass_std_scaler.pkl`: (Quasi) standard scaler in `pickle` format.
7. `mass_std_scaler.json`: JSON version of (quasi) standard scaler with plain-valued parameters.  Notice that `"coord"`, `"dist"`, and `"phi"` means and standard deviations are given for `h`-normalized features.
8. `Preprocessing.py`: Customized feature-type-based standardization transformer class.
