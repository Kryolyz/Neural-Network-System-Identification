# Neural-Network-System-identification
A simple script for system identification based on tensorflows neural network. It uses Tensorflows neural nets to make a simple Multi-Input-Multi-Output regression. The concept is basically identical to regular least squares regression methods. It also has the same weaknesses which include the inability to find nonlinearities itself. Theoretically a neural net should be able to identify any system using only 2 layers (input and output) as long as all nonlinearities are considered beforehand, so the code just needs minor adjustments (change input and output to what you want and calculate nonlinearites before sending the variables into the network) if you want to use it.

# References
I included a bibtex-file in the References folder which contains all the material I used. 
