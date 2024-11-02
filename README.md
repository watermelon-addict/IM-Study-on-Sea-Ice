[Hidden comment]: # 

### In this study, we combine the 100 year old Ising model in classical statistical physics with modern deep learning methods to examine Arctic sea ice dynamics, a crucial indicator of climate change. Upon the binary-spin Ising setup, we introduce continuous spin values, which capture the real-world ice/water phase transition, and an innovative inertia factor, which represents the natural resistance to state changes. The generalized model is utilized for the Monte Carlo simulation of the sea ice evolution in a focus area of the Arctic region, by engaging the Metropolis-Hastings algorithm and training three deep learning models: a simple convolutional neural network, a much deeper residual network, and a vision transformer, to solve the inverse Ising problem. Using the sea ice concentration data collected by the National Snow and Ice Data Center, our model proves to have strong explanatory power. The simulated configurations exhibit striking similarity with the actual ice/water images, and two numerical measures calculated from the simulation results—the ice coverage percentage and the ice extent—match closely with the data statistics. Moreover, the Ising parameters predicted by the convolutional neural network demonstrate the substantial impact of the external forces, which can be further enriched and linked to the environmental factors in other global warming analyses. This study identifies the fundamental physical mechanism governing sea ice dynamics. It also validates the vast potential of pairing classical physics with cutting-edge technologies in climate science, thereby presenting ample possibilities for future interdisciplinary research.

<br/><br/>
# 1. Introduction
<br/>

This study models Arctic Sea Ice dynamics at a large scale by innovating on the centennial Ising model (IM). Rapid loss of the Arctic sea ice over the past four decades [1] has been an alarming phenomenon that points to drastic global warming, a pressing challenge that calls for collective actions by the entire humankind. To accurately predict the Arctic sea ice evolution, which helps prepare for the subsequent environmental and economic impact, therefore has become an urgent task for researchers across diverse disciplines. The fact that 2023 recorded the hottest year in history [2] and 2024 may break the temperature record again, adds even greater severity to such urgency. As an endeavor to fulfill this task, this study innovates upon the classical Ising Model in statistical physics and trains Convolutional Neural Network and Vision Transformer models in Deep Learning to solve for Ising interaction parameters. By delivering an excellent match between model simulations and actual observations, this study unleashes the power of coupling classical physics with deep learning technologies in climate change research and other interdisciplinary studies.
<br/>
This  research report starts with a brief review on Ising model and neural network literature, and an introduction of the significance of Arctic sea ice in Section 1; Section 2 lays out the theoretical framework of our generalized Ising model and the neural networks; Section 3 describes the Arctic sea ice data; Section 4 illustrates the computational setups; Section 5 presents the results and analysis, followed by discussions in Section 6 at the end. A previous paper of this research was published on Journal of Applied Physics [3] recently; this research report is an enhancement based on that paper.

<br/>
1.1	Ising model
<br/>
The classical Ising model (IM) is the backbone of this study. It was first formalized by physicists Ernst Ising and Wilhelm Lenz to explain the equilibrium and phase transition in magnetic systems. The one-dimensional (1-D) IM was solved by Ising in his 1924 thesis [4] [5] [6], which proves the non-existence of phase transition in the 1-D IM. In 1944, Lars Onsager [7] was able to solve the two-dimensional (2-D) square-lattice IM analytically. Contradictory to the 1-D case, Onsager identified that there exists a critical temperature Tc = 2.27 J/kB when the phase transition happens in a 2-D IM. Later studies of IM in higher dimensions have been closely associated with various developments in advanced 20th-century physics and mathematical theories, including the transfer-matrix method [8] [9], quantum field theory [10], mean-field theory [11], etc.
<br/>
Over the years, the IM has found wide success beyond physics. Specifically, the Kinetic IM [11] [12] [13], built upon the equilibrium version, has been proposed to analyze biology, environmental science, machine learning [14] [15], social science, and economic and financial systems. These applications are usually implemented as a discrete time Markov chain of the spin lattice, with spin interactions bounded to finite distance. In biology and neuroscience, IM applications include but are not limited to the condensation of DNA [16], genetics [17], neural networks [18] [19], neuron spike [20], neuron activity in cell assemblies [21], and ligands to receptors binding in cells [22]. In environmental science, the IM has been employed to investigate land pattern dynamics [23] [24]. A few years ago, Ma, Sudakov, Strong and Golden have successfully used the 2-D IM to capture the essential mechanism of the ice melt ponds equilibrium configuration [25]. In social science and economics, the IM has been applied to research in urban segregation [26], crisis study [27], stability of money [28], etc.

<br/><br/>
1.2	Deep learning with convolutional neural networks and transformers
<br/>
My study falls into a broad body of literature that employs artificial neural networks (ANNs), or neural networks (NNs), a branch of artificial intelligence and machine learning inspired by the structure and functioning of the human brain [29] [30]. Most modern deep learning models are based on multi-layered ANNs. Interestingly, Ising model is considered as the first non-learning recurrent neural network (RNN) architecture [31], which laid the foundation for the 2024 Nobel Prize winning Hopfield network [32].
<br/>
The convolutional neural network (CNN) [33] [34] [35] employed in this study is a specialized type of NNs used to analyze data with grid-like topology, which revolutionized computer vision in the 2010s [31] [33]. CNNs have gained much success in image and video recognition [36], and also been widely applied to time series analysis [37], recommender systems [38], natural language processing [39], etc. The rapid development of CNN led to a series of state-of-the-art (SOTA) models, including AlexNet in 2012 [40], VGG in 2014 [41], InceptionNet/GoogleNet in 2014 [42], ResNet in 2015 [43], DenseNet in 2016 [44], EfficientNet in 2019 [45], etc.
<br/>
Transformer is a groundbreaking deep learning architecture proposed by Google in the 2017 paper “Attention is all you need” [46]. Since then, it has rapidly surpassed RNN such as LSTM [47]/GRU [48] to become the SOTA model in Natural Language Processing (NLP). Almost all of the current Large Language Models (LLMs) are based on the transformer architecture, including Generative Pretrained Transformers (GPT) [49] [50] developed by OpenAI, Bidirectional Encoder Representations from Transformers (BERT) [51] by Google, Large Language Model Meta AI (Llama) [52] by Meta, etc. Transformers have also found wide applications besides NLP, including playing chess [53] and go [54], multi-modal processing [55], computer vision [56] [57], and many more. Specifically, a vision transformer (ViT) model was proposed in 2021 [57] and achieved excellent performance at image classification compared to CNN.
The images of 2-D IM lattices are well-qualified candidates for CNN and ViT, which are explored in this study. The architecture of CNN and ViT models will be described in Section 4.
<br/><br/>

1.3	Arctic sea ice
<br/>
The reversible phase transition between water and ice makes the IM a great tool to study the dynamics of a surface region with the co-existence of both states. In this study, we apply a 2-D IM lattice to study the dynamics of Arctic sea ice melting and freezing cycles, a major climate change indicator that is of significant environmental, economic and social significance [58].
<br/>
Sea ice is undoubtedly an integral part of the Arctic Ocean and the earth. In the dark winter months, ice covers almost the entirety of the Arctic Ocean, and the ice extent—defined as the percentage of the areas that are covered by at least 15% ice—and the ice thickness typically reaches its peak around March. Starting in late spring, ice melting gradually exceeds water freezing due to higher temperatures and longer hours of sunlight exposure. Sea ice typically reaches the minimum extent and thickness in mid-September, when ice coverage can drop to under half of the winter maximum [34]. After mid-September, sea water freezing starts to exceed ice melting, so ice coverage expands. This cycle repeats annually.
<br/>

Albedo, the percentage of incident light reflected from the surface of the earth, is highly dependent on the ice extent [59]. Light-colored ice or snow reflects more light than blue-colored liquid water; therefore, ice is essential to keeping the Arctic at a cooler temperature and subsequently maintaining the energy balance around the globe. If the energy balance is broken, as ice decline has been detected in recent years, the ice-albedo feedback loop effect may occur, i.e., less reflection and more absorption of solar energy, leading to even more ice loss and further global warming. Moreover, the Arctic ecosystem is inversely impacted by the decline in sea ice coverage, which, for instance, threatens the lives of polar bears and walruses who rely on sea ice for hunting and breeding [60]. 
<br/>

Data recorded by the National Aeronautics and Space Administration (NASA) and the National Snow and Ice Data Center (NSIDC) since 1979 has shown substantial declines in both ice extent and thickness in the Arctic, despite the year-over-year fluctuations in either direction. The lowest Arctic sea ice extent was observed in September of 2012 [1] [61]; between 2013 and 2022, the ice extent was higher than the 2012 minimum, but still much lower than the average of the past four decades. 2023 has recorded the hottest year by a significant margin so far [2] [62], and 2024 might break the record again [63]. Some questions then come to us naturally: how does the Artic sea ice extent in the most recent years compare to the 2012 level? Can our model simulations closely match the observations in the real data? These questions will be addressed in Section 5 and 6.
<br/><br/>

# 2	Theoretical framework
<br/>
2.1	Classical Ising model
<br/>
The system described by an IM is a set of lattice sites, each having a spin that interacts with its neighbors. The Hamiltonian function [4] for the lattice σ in a standard IM is given as <br/>
<figure>
    <img src="https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/b9cbaf41-2590-46f0-9473-45629398363d)" width="250" height="50">
</figure>
<br/>
where σ_i represents the spin variables at site i and takes the value of +1 or -1; J_ij represents the interaction between sites i and j and can take positive values for ferromagnetic and paramagnetic materials, or negative for antiferromagnetic materials; B_i captures the interaction between the external field and site i. i and j range across the full lattice, which can be one, two or higher dimensions, and <i, j> represents pairs of spins at sites i and j that interact with each other. In a simple setup, each spin may only interact with its nearest neighbors, so <i, j> sums over adjacent sites only. For example, in a simple 2-D IM, each spin interacts only with the sites positioned immediately left, right, above, and below. 
<br/>
In statistical physics, the configuration probability of lattice σ follows the Boltzmann distribution 
<br/>
    ![image](https://github.com/user-attachments/assets/5f266ac7-cf2c-4816-80d6-300fd73bac70)
<br/>
<figure>
    <img src="https://github.com/user-attachments/assets/5f266ac7-cf2c-4816-80d6-300fd73bac70)" width="250" height="50">
</figure>
<br/>
where Z is the partition function: <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/bde41b65-a38b-4ad6-a56f-61f525a37bf4)

and <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/86e2baa8-fcec-4fbb-97d2-96a360444b21)
<br/>
β is the inverse temperature; 
$k_B$ is the Boltzmann constant; T is the IM temperature, which differentiates from the ambient temperature discussed later.
<br/>
The evolution of the kinetic IM runs through a series of spin flips over the lattice. The probability of each spin flip depends on whether such a flip increases or reduces the Hamiltonian of the system. Mathematically the probability is determined by $min⁡(1,e^{(-β(H_ν-H_μ ) )})$ [64], where Hv and Hµ represent the Hamiltonian of the system before and after the flip. It can be easily seen that higher IM temperatures lead to greater thermal fluctuations and larger variances in the spin value distribution, while lower IM temperatures result in fewer fluctuations.

<br/><br/>
Most studies of the IM focus on binary values of the spins, i.e., σ_i takes values of +1 or -1 only. However, the sea ice data for each location takes varying values between 0 and 1 that represent the percentage of ice coverage. Therefore, we generalize the IM to allow for continuous spin values that can take any real number between -1 and +1.
<br/>

Moreover, we incorporate an innovative inertia factor I, and the probability of each flip is determined by  <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/d2ce424e-7772-4a88-9584-ca15272a8c84)

The newly added ![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/750d0eff-7b45-4cba-b36e-ca77d7ca9590)
 accounts for the energy needed to overcome the inertia of the spin change.
<br/><br/>

<br/>
<br/>

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


the configuration probability of lattice σ follows the Boltzmann distribution <br/>



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


