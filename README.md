# Neural-Network-System-Identification
A simple script for system identification based on tensorflows neural networks. It uses Tensorflows neural nets to make a simple Multi-Input-Multi-Output regression with data generated from a state-space. I only needed it for a quick proof-of-concept for college, so it doesn't include reading data files etc. The concept is basically identical to regular least squares regression methods. It also has the same weaknesses which include the inability to find nonlinearities itself. Theoretically a neural net should be able to identify any system using only 2 layers (input and output) as long as all nonlinearities are considered beforehand, so the code just needs minor adjustments (change input and output to what you want and calculate nonlinearites before sending the variables into the network) if you want to use it.

The as-is version of the file identifies a DC-Motor state-space and prints out the weights and biases of the network, plots a graph showing the difference between the predicted value of the network and the real value of the state space, plots the loss as a function of the iterations and shows a visualization of the network.

Also i'm kind of a messy coder so the file barely contains any comments (and the existing ones are german), however if you have some experience with python and tensorflow everything should be ok.

# Requirements
The Python packages required are tensorflow, numpy, matplotlib and networkx. To install using pip or conda, use exactly those names.
Example : pip install tensorflow
OR      conda install tensorflow

# References
I included a bibtex-file in the References folder which contains all the papers I used. 
