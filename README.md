# ECNet
### Error-correcting neural network for semi-Lagrangian advection in the level-set method

We include the following files:
1. `requirements.txt`: Contains `python` virtual environment packages to load the neural network in TensorFlow/Keras.
2. `mass_nnet.h5`: Tensorflow/Keras model with no optimizer.
3. `mass_nnet.json`: JSON version of neural network with hidden-layer weights encoded in `Base64` but represented as `ASCII` text.  It does not include the `h`-denormalization operation that comes at the end of `mass_nnet.h5`.
4. `mass_pca_scaler.pkl`: PCA scaler stored in `pickle` format.
5. `mass_pca_scaler.json`: JSON version of PCA scaler with plain-valued parameters.
6. `mass_std_scaler.pkl`: (Quasi) standard scaler in `pickle` format.
7. `mass_std_scaler.json`: JSON version of (quasi) standard scaler with plain-valued parameters.  Notice that `"coord"`, `"dist"`, and `"phi"` means and standard deviations are given for `h`-normalized features.
8. `Preprocessing.py`: Customized feature-type-based standardization transformer class.
