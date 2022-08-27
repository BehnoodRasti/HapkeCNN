# HapkeCNN
Blind Nonlinear Unmixing for Intimate Mixtures Using Hapke Model and Convolutional Neural Network

If you use this code and/or our simulated datasets please do not forget cite the following paper: B. Rasti, B. Koirala, and P. Scheunders, "HapkeCNN: Blind Nonlinear Unmixing for Intimate Mixtures Using Hapke Model and Convolutional Neural Network," in IEEE Transactions on Geoscience and Remote Sensing.

HapkeCNN is a blind nonlinear unmixing technique for intimate mixtures using the Hapke model and convolutional neural networks. We use the Hapke model and a fully convolutional encoder-decoder deep network for the nonlinear unmixing. Additionally, we propose a novel loss function that includes three terms; 1) a quadratic term
based on the Hapke model, that captures the nonlinearity, 2) the reconstruction error of the reflectances, to ensure the fidelity of the reconstructed reflectance, and 3) a minimum volume total variation term that exploits the geometrical information to estimate the endmembers in the absence of pure pixels in the hyperspectral data. The proposed method is evaluated using two simulated and two real datasets. We compare the results of endmember and abundance estimation with a number of nonlinear, and projection-based linear unmixing techniques. The experimental results confirm that HapkeCNN considerably outperforms the state-of-the-art nonlinear approaches

We provided all the datasets and the ground references used in the manuscript. The noisy datasets are for 30 dB. Below you see the results for Simulated dataset 2. 



![image](https://user-images.githubusercontent.com/61419984/187027403-0ae45cd1-a5fe-4db1-a97f-6ff9300f0edd.png)![image](https://user-images.githubusercontent.com/61419984/187027352-e78d5dbc-ef19-4ab1-b618-a223539d614c.png)


