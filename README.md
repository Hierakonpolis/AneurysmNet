# AneurysmNet
![logos](logos.png)
On this page you can find our implementation of the aneurysm segmentation algorithm we submitted for the [ADAM challenge](http://adam.isi.uu.nl/) in MICCAI 2020. 
Our submission ranked 4th for the aneurysm detection challenge, with a sensitivity of 0.60 and a false positive rate of 0.36, and 3rd for the aneurysm segmentation challenge (Dice 0.28, Mean Hausdorff distance 18.13, Volume similarity	0.39). See the full ranking [here](http://adam.isi.uu.nl/results/results-miccai-2020/) and the docker containers [here](http://adam.isi.uu.nl/results/results-miccai-2020/participating-teams-miccai-2020/).

Our approach is based on an ensamble of 18 networks, trained using 3 different architectures/loss functions using 6 different validation sets. More information and aknowledgements will later be added to this page.
