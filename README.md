[Hidden comment]: # 

### In this study, we combine the 100 year old Ising model in classical statistical physics with modern deep learning methods to examine Arctic sea ice dynamics, a crucial indicator of climate change. Upon the binary-spin Ising setup, we introduce continuous spin values, which capture the real-world ice/water phase transition, and an innovative inertia factor, which represents the natural resistance to state changes. The generalized model is utilized for the Monte Carlo simulation of the sea ice evolution in a focus area of the Arctic region, by engaging the Metropolis-Hastings algorithm and training three deep learning models: a simple convolutional neural network, a much deeper residual network, and a vision transformer, to solve the inverse Ising problem. Using the sea ice concentration data collected by the National Snow and Ice Data Center, our model proves to have strong explanatory power. The simulated configurations exhibit striking similarity with the actual ice/water images, and two numerical measures calculated from the simulation results—the ice coverage percentage and the ice extent—match closely with the data statistics. Moreover, the Ising parameters predicted by the convolutional neural network demonstrate the substantial impact of the external forces, which can be further enriched and linked to the environmental factors in other global warming analyses. This study identifies the fundamental physical mechanism governing sea ice dynamics. It also validates the vast potential of pairing classical physics with cutting-edge technologies in climate science, thereby presenting ample possibilities for future interdisciplinary research.

<br/><br/>
# 1. Introduction
<br/>

This study models Arctic Sea Ice dynamics at a large scale by innovating on the centennial Ising model (IM). Rapid loss of the Arctic sea ice over the past four decades [1] has been an alarming phenomenon that points to drastic global warming, a pressing challenge that calls for collective actions by the entire humankind. To accurately predict the Arctic sea ice evolution, which helps prepare for the subsequent environmental and economic impact, therefore has become an urgent task for researchers across diverse disciplines. The fact that 2023 recorded the hottest year in history [2] and 2024 may break the temperature record again, adds even greater severity to such urgency. As an endeavor to fulfill this task, this study innovates upon the classical Ising Model in statistical physics and trains Convolutional Neural Network and Vision Transformer models in Deep Learning to solve for Ising interaction parameters. By delivering an excellent match between model simulations and actual observations, this study unleashes the power of coupling classical physics with deep learning technologies in climate change research and other interdisciplinary studies.
This  research report starts with a brief review on Ising model and neural network literature, and an introduction of the significance of Arctic sea ice in Section 1; Section 2 lays out the theoretical framework of our generalized Ising model and the neural networks; Section 3 describes the Arctic sea ice data; Section 4 illustrates the computational setups; Section 5 presents the results and analysis, followed by discussions in Section 6 at the end. A previous paper of this research was published on Journal of Applied Physics [3] recently; this research report is an enhancement based on that paper.
1.1	Ising model
The classical Ising model (IM) is the backbone of this study. It was first formalized by physicists Ernst Ising and Wilhelm Lenz to explain the equilibrium and phase transition in magnetic systems. The one-dimensional (1-D) IM was solved by Ising in his 1924 thesis [4] [5] [6], which proves the non-existence of phase transition in the 1-D IM. In 1944, Lars Onsager [7] was able to solve the two-dimensional (2-D) square-lattice IM analytically. Contradictory to the 1-D case, Onsager identified that there exists a critical temperature Tc = 2.27 J/kB when the phase transition happens in a 2-D IM. Later studies of IM in higher dimensions have been closely associated with various developments in advanced 20th-century physics and mathematical theories, including the transfer-matrix method [8] [9], quantum field theory [10], mean-field theory [11], etc.
Over the years, the IM has found wide success beyond physics. Specifically, the Kinetic IM [11] [12] [13], built upon the equilibrium version, has been proposed to analyze biology, environmental science, machine learning [14] [15], social science, and economic and financial systems. These applications are usually implemented as a discrete time Markov chain of the spin lattice, with spin interactions bounded to finite distance. In biology and neuroscience, IM applications include but are not limited to the condensation of DNA [16], genetics [17], neural networks [18] [19], neuron spike [20], neuron activity in cell assemblies [21], and ligands to receptors binding in cells [22]. In environmental science, the IM has been employed to investigate land pattern dynamics [23] [24]. A few years ago, Ma, Sudakov, Strong and Golden have successfully used the 2-D IM to capture the essential mechanism of the ice melt ponds equilibrium configuration [25]. In social science and economics, the IM has been applied to research in urban segregation [26], crisis study [27], stability of money [28], etc.

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

### Ising model
<br/>
The Hamiltonian function for the lattice σ in a standard IM is given as: <br/>
<figure>
    <img src="https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/b9cbaf41-2590-46f0-9473-45629398363d)" width="250" height="50">
</figure>

the configuration probability of lattice σ follows the Boltzmann distribution <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/ca5f010f-1672-4d0d-b045-2ffb32348df5)

where Z is the partition function: <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/bde41b65-a38b-4ad6-a56f-61f525a37bf4)

and <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/86e2baa8-fcec-4fbb-97d2-96a360444b21)
<br/><br/>

Most studies of the IM focus on binary values of the spins, i.e., σ_i takes values of +1 or -1 only. However, the sea ice data for each location takes varying values between 0 and 1 that represent the percentage of ice coverage. Therefore, we generalize the IM to allow for continuous spin values that can take any real number between -1 and +1.
<br/>

Moreover, we incorporate an innovative inertia factor I, and the probability of each flip is determined by  <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/d2ce424e-7772-4a88-9584-ca15272a8c84)

The newly added ![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/750d0eff-7b45-4cba-b36e-ca77d7ca9590)
 accounts for the energy needed to overcome the inertia of the spin change.
<br/><br/>

### Ising parameter setup
𝐽_𝑖𝑗 and I set to be constant each period<br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/1f44feee-9413-49b0-8902-5872157912ab)
<br/>
𝛽=1
<br/>

### Metropolis Markov Chain Monte Carlo (MCMC) simulation:
In our study, we follow the Metropolis MCMC process for the simulation of the IM lattice evolution:
<figure>
    <img src="https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/eac18e98-3770-4736-9d9c-9d3adf0a1edf" width="500" height="400">
</figure>
<br/><br/>

# Convolutional Neural Network

The inverse Ising problem: given the start and end state images of the Ising lattices, how do we determine the IM interaction parameters (J, B, I)? In this study, we will train a Convolutional Neural Network (CNN) deep learning model for this task.
<br/>

The architecture of our CNN is illustrated in Figure 2. The total number of trainable parameters stays at 213,101, making this a relatively small deep learning algorithm that can be trained on the CPU of a personal computer.
<br/>
![CNN_Architecture](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/a573a168-5530-4ee0-9103-0bad447cb3cc)
*Figure 2: CNN Architecture*
<br/>

<br/><br/>

The training data for this CNN are generated following the Metropolis MCMC simulation steps described previously. To be specific, we start with the Ising lattice at the initial state of a simulation period and randomly select 10,000 set of parameters (J, B_0,B_x, B_y, I); for each set of parameters, we run the Metropolis simulation steps. As a result, we generate 10,000 sets of training samples corresponding to each of the initial states. We combine such training data for the full year as the input for CNN training.
<br/><br/>

# Results

### Simulation results for 2022

Figure 3 shows the semi-monthly NRTSI sea ice images in the focus area from June 16th, 2022 to Jan 1st, 2023. <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/3d981b64-3b10-42e9-a6e9-709d24295919)
<br/>
*Figure 3: The actual semi-monthly sea ice evolution in the focus area in 2022*
<br/><br/>

The CNN-predicted Ising parameters (J, B_0,B_x, B_y, I) for each simulation period in 2022 are shown in Table 1. <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/2e80a462-0299-4775-9786-4b5ccb5d822a)
<br/>
*Table 1: CNN predicted Ising parameters for the 2022 sea ice evolution*
<br/><br/>

The simulated sea ice images for each 2022 period are shown in Figure 4 utilizing the CNN predicted Ising parameters in Table 1. These images exhibit excellent similarity to Figure 3, demonstrating the strong explanatory power of our Ising model.  <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/584d70e9-a89f-4de2-9a1a-7750f7baf1ec)
<br/>
*Figure 4: The simulated semi-monthly sea ice evolution in the focus area in 2022*
<br/><br/>


The differences in ice coverages across the entire focus area for each of the simulation period in Figure 3 and Figure 4 are calculated; the results are illustrated as the heatmaps in Figure 5. <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/b8f58896-f807-4cf8-bb93-91783beb31a8)
<br/>
*Figure 5: Heatmaps illustrating the absolute difference (between 0 and 1) in ice coverages between Figure 3 and Figure 4 for each semi-monthly period*
<br/><br/>

Figure 6 shows two key numerical measures based on both actual observations and the simulation results for our focus area: the ice coverage percentage, i.e., the mean of the ice coverage across the entire lattice, and the ice extent, i.e., the percentage of areas that are covered by at least 15% ice. <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/0a1e7d88-6dca-4edd-88f4-bd73f6556f4b)
<br/>
*Figure 6: (a) The ice coverage percentage in the focus area in 2022. (b) The ice extent for the same period. Blue curves are the actual measures from the NRTSI data; orange ones show the IM simulation results.*
<br/><br/>

### Other results

More results, including the daily sea ice evolution, and simulation resutls for 2023 and other years, can be found in: E. Wang, "Deep Learning on a Novel Ising Model to Study Arctic Sea Ice Dynamics", 2024. <br/>

<br/><br/>
# Code files
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
E. Wang, "Deep Learning on a Novel Ising Model to Study Arctic Sea Ice Dynamics", 2024. <br/>
E. Ising, "Beitrag zur Theorie des Ferromagnetismus," Z. Phys, vol. 31, no. 1, p. 2530258, 1925. <br />
Y.-P. Ma, I. Sudakov, C. Strong and K. Golden, "Ising model for melt ponds on Arctic sea ice," New Journal of Physics, vol. 21, p. 063029, 2019. <br />
M. Krasnytska, B. Berche, Y. YuHolovatch and R. Kenna, "Ising model with variable spin/agent strengths," Journal of Physics: Complexity, vol. 1, p. 035008, 2020.
J. Albert and R. H. Swendsen, "The Inverse Ising Problem," Physics Procedia, vol. 57, pp. 99-103, 2014.<br />
N. Walker, K. Tam and M. Jarrell, "Deep learning on the 2‑dimensional Ising model to extract the crossover region with a variational autoencoder," Scientific Reports, vol. 10, p. 13047, 2020. <br />
W. N. Meier, J. S. Stewart, H. Wilcox, M. A. Hardman and S. D. J., "Near-Real-Time DMSP SSMIS Daily Polar Gridded Sea Ice Concentrations, Version 2," NASA National Snow and Ice Data Center Distributed Active Archive Center, Boulder, Colorado USA, 2023.<br />



<br/><br/>
# Copyright from NSIDC on using their sea ice data and code:

Copyright (c) 2023 Regents of the University of Colorado
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.


