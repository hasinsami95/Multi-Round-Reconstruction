# Multi-Round-Reconstruction
The user needs to download the CIFAR-10 and FEMNIST dataset and save them in a folder named "data", which will be used in the code.

Our implementation uses the "inversefed" folder available at https://github.com/JonasGeiping/invertinggradients for gradient inversion attack.

To ensure same results all the time, the reconstruction attack was performed on a saved resnet-18 model named "Model.pt".

In the code- "multi_round_attack.ipynb",after local training, we partition the entire gradient into 6 segments and create matrix "A" and gradient sum for each segment separately to allow parallel computation.

We execute 500 rounds of training, and save our results A_conc1, A_conc2, A_conc3, A_conc4, A_conc5, A_conc6 in 6 matrices ("A_conc1.npy", "A_conc2.npy", "A_conc3.npy", "A_conc4.npy", "A_conc5.npy", "A_conc6.npy"). Note that these matrices will be used to generate matrix "A" defined in the paper for each coordinate within each segment.

We save our results err1, err2, err3, err4, err5, err6 in six matrices ("agg1.npy", "agg2.npy", "agg3.npy", "agg4.npy","agg5.npy", "agg6.npy") each corresponding to sparse gradient aggregate over 500 rounds for 1/6th of the entire gradient.

We save the error accumulation numbers in "T.npy".

We can repeat the process for more rounds if needed.

The codes "A_matrix_11_concatenated.ipynb", "A_matrix_22_concatenated.ipynb", "A_matrix_33_concatenated.ipynb", "A_matrix_44_concatenated.ipynb", "A_matrix_55_concatenated.ipynb", "A_matrix_66_concatenated.ipynb" are written to construct the matrix "A" for each coordinate and perform least square solution to recover the gradients in each of those 6 segments.
