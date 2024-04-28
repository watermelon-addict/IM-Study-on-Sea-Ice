[Hidden comment]: # 

### In this study, we combine the 100 year old Ising model in classical statistical physics with modern deep learning methods to examine Arctic sea ice dynamics, a crucial indicator of climate change. Upon the binary-spin Ising setup, we introduce continuous spin values, which capture the real-world ice/water phase transition, and an innovative inertia factor, which represents the natural resistance to state changes. The generalized model is utilized for the Monte Carlo simulation of the sea ice evolution in a focus area of the Arctic region, by engaging the Metropolis-Hastings algorithm and training a convolutional neural network to solve the inverse Ising problem. Using the sea ice concentration data collected by the National Snow and Ice Data Center, our model proves to have strong explanatory power. The simulated configurations exhibit striking similarity with the actual ice/water images, and two numerical measures calculated from the simulation results—the ice coverage percentage and the ice extent—match closely with the data statistics. 

<br/><br/>
# Data description
<br/>

Our study uses the “Near-Real-Time DMSP SSMIS Daily Polar Gridded Sea Ice Concentrations” (NRTSI) dataset collected by the National Snow and Ice Data Center (NSIDC), as shown in Figure 1 (a). For this research paper, we focus on studying a specific geographic region bounded by the black square. A zoom-in image of this focus area is shown in Figure 1(b) . The area contains 60 rows and 60 columns in the data file, covering approximately 1500km x 1500km.

<!---
!["Figure 1: Arctic sea ice and the focus area"](/images/Figure1.jpg)
-->

<figure>
    <img src="/images/Figure1.jpg" width="400" height="250">
    <figcaption> Figure 1: (a) Arctic sea ice NRTSI image on Sept 16, 2022 and (b) the focus area of 1,500km X 1,500km </figcaption>
</figure>


<br/><br/>
# Continuous spin Ising model with inertia factor
<br/>
The Hamiltonian function for the lattice σ in a standard IM is given as:
<figure>
    <img src="https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/b9cbaf41-2590-46f0-9473-45629398363d)" width="250" height="50">
</figure>

the configuration probability of lattice σ follows the Boltzmann distribution
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/ca5f010f-1672-4d0d-b045-2ffb32348df5)

where Z is the partition function:
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/bde41b65-a38b-4ad6-a56f-61f525a37bf4)

and
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/86e2baa8-fcec-4fbb-97d2-96a360444b21)
<br/><br/>


<br/><br/>
# Convolutional Neural Network
<br/>

![CNN_Architecture](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/a573a168-5530-4ee0-9103-0bad447cb3cc)
<br/><br/>
*Figure 2: CNN Architecture*

<br/><br/>

The code files this project uses when employing the noval Continuous Spin Ising Model and Convolutional Neural Networks to study the dynamics of Arctic Sea Ice include:  
•	download_NSIDC.py<br/>
•	ReadSeaIce.py<br/>
•	IIMConstants.py<br/>
•	IceIsingCont.py<br/>
•	IIMCNNModel.py<br/>
•	IIMCNNRun.py<br/>
•	IIM_CNN_Results.py<br/>
•	IIMSimul.py<br/>
•	Gen_Figures.py.<br/>
<br/>
Detailed desription of the code is in CodeList.txt
<br/><br/>


# Key references: <br/>
E. Ising, "Beitrag zur Theorie des Ferromagnetismus," Z. Phys, vol. 31, no. 1, p. 2530258, 1925. <br />
Y.-P. Ma, I. Sudakov, C. Strong and K. Golden, "Ising model for melt ponds on Arctic sea ice," New Journal of Physics, vol. 21, p. 063029, 2019. <br />
M. Krasnytska, B. Berche, Y. YuHolovatch and R. Kenna, "Ising model with variable spin/agent strengths," Journal of Physics: Complexity, vol. 1, p. 035008, 2020.
J. Albert and R. H. Swendsen, "The Inverse Ising Problem," Physics Procedia, vol. 57, pp. 99-103, 2014.<br />
N. Walker, K. Tam and M. Jarrell, "Deep learning on the 2‑dimensional Ising model to extract the crossover region with a variational autoencoder," Scientific Reports, vol. 10, p. 13047, 2020. <br />
W. N. Meier, J. S. Stewart, H. Wilcox, M. A. Hardman and S. D. J., "Near-Real-Time DMSP SSMIS Daily Polar Gridded Sea Ice Concentrations, Version 2," NASA National Snow and Ice Data Center Distributed Active Archive Center, Boulder, Colorado USA, 2023.<br />



<br/><br/>
<br/><br/>
# Copyright for download_NSIDC.py:
<br/><br/>
Copyright (c) 2023 Regents of the University of Colorado
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.


