[Hidden comment]: # 

### In this study, we combine the 100 year old Ising model in classical statistical physics with modern deep learning methods to examine Arctic sea ice dynamics, a crucial indicator of climate change. Upon the binary-spin Ising setup, we introduce continuous spin values, which capture the real-world ice/water phase transition, and an innovative inertia factor, which represents the natural resistance to state changes. The generalized model is utilized for the Monte Carlo simulation of the sea ice evolution in a focus area of the Arctic region, by engaging the Metropolis-Hastings algorithm and training three deep learning models: a simple convolutional neural network, a much deeper residual network, and a vision transformer, to solve the inverse Ising problem. Using the sea ice concentration data collected by the National Snow and Ice Data Center, our model proves to have strong explanatory power. The simulated configurations exhibit striking similarity with the actual ice/water images, and two numerical measures calculated from the simulation results—the ice coverage percentage and the ice extent—match closely with the data statistics. Moreover, the Ising parameters predicted by the convolutional neural network demonstrate the substantial impact of the external forces, which can be further enriched and linked to the environmental factors in other global warming analyses. This study identifies the fundamental physical mechanism governing sea ice dynamics. It also validates the vast potential of pairing classical physics with cutting-edge technologies in climate science, thereby presenting ample possibilities for future interdisciplinary research.

<br/><br/>
# 1. Introduction
<br/>

This study models Arctic Sea Ice dynamics at a large scale by innovating on the centennial Ising model (IM). Rapid loss of the Arctic sea ice over the past four decades [1] has been an alarming phenomenon that points to drastic global warming, a pressing challenge that calls for collective actions by the entire humankind. To accurately predict the Arctic sea ice evolution, which helps prepare for the subsequent environmental and economic impact, therefore has become an urgent task for researchers across diverse disciplines. The fact that 2023 recorded the hottest year in history [2] and 2024 may break the temperature record again, adds even greater severity to such urgency. As an endeavor to fulfill this task, this study innovates upon the classical Ising Model in statistical physics and trains Convolutional Neural Network and Vision Transformer models in Deep Learning to solve for Ising interaction parameters. By delivering an excellent match between model simulations and actual observations, this study unleashes the power of coupling classical physics with deep learning technologies in climate change research and other interdisciplinary studies.
<br/>
This  research report starts with a brief review on Ising model and neural network literature, and an introduction of the significance of Arctic sea ice in Section 1; Section 2 lays out the theoretical framework of our generalized Ising model and the neural networks; Section 3 describes the Arctic sea ice data; Section 4 illustrates the computational setups; Section 5 presents the results and analysis, followed by discussions in Section 6 at the end. A previous paper of this research was published on Journal of Applied Physics [3] recently; this research report is an enhancement based on that paper.

<br/>
<h2> 1.1  Ising model </h2>
<br/>
The classical Ising model (IM) is the backbone of this study. It was first formalized by physicists Ernst Ising and Wilhelm Lenz to explain the equilibrium and phase transition in magnetic systems. The one-dimensional (1-D) IM was solved by Ising in his 1924 thesis [4] [5] [6], which proves the non-existence of phase transition in the 1-D IM. In 1944, Lars Onsager [7] was able to solve the two-dimensional (2-D) square-lattice IM analytically. Contradictory to the 1-D case, Onsager identified that there exists a critical temperature Tc = 2.27 J/kB when the phase transition happens in a 2-D IM. Later studies of IM in higher dimensions have been closely associated with various developments in advanced 20th-century physics and mathematical theories, including the transfer-matrix method [8] [9], quantum field theory [10], mean-field theory [11], etc.
<br/>
Over the years, the IM has found wide success beyond physics. Specifically, the Kinetic IM [11] [12] [13], built upon the equilibrium version, has been proposed to analyze biology, environmental science, machine learning [14] [15], social science, and economic and financial systems. These applications are usually implemented as a discrete time Markov chain of the spin lattice, with spin interactions bounded to finite distance. In biology and neuroscience, IM applications include but are not limited to the condensation of DNA [16], genetics [17], neural networks [18] [19], neuron spike [20], neuron activity in cell assemblies [21], and ligands to receptors binding in cells [22]. In environmental science, the IM has been employed to investigate land pattern dynamics [23] [24]. A few years ago, Ma, Sudakov, Strong and Golden have successfully used the 2-D IM to capture the essential mechanism of the ice melt ponds equilibrium configuration [25]. In social science and economics, the IM has been applied to research in urban segregation [26], crisis study [27], stability of money [28], etc.

<br/><br/>
<h2> 1.2	Deep learning with convolutional neural networks and transformers </h2>
<br/>
My study falls into a broad body of literature that employs artificial neural networks (ANNs), or neural networks (NNs), a branch of artificial intelligence and machine learning inspired by the structure and functioning of the human brain [29] [30]. Most modern deep learning models are based on multi-layered ANNs. Interestingly, Ising model is considered as the first non-learning recurrent neural network (RNN) architecture [31], which laid the foundation for the 2024 Nobel Prize winning Hopfield network [32].
<br/>
The convolutional neural network (CNN) [33] [34] [35] employed in this study is a specialized type of NNs used to analyze data with grid-like topology, which revolutionized computer vision in the 2010s [31] [33]. CNNs have gained much success in image and video recognition [36], and also been widely applied to time series analysis [37], recommender systems [38], natural language processing [39], etc. The rapid development of CNN led to a series of state-of-the-art (SOTA) models, including AlexNet in 2012 [40], VGG in 2014 [41], InceptionNet/GoogleNet in 2014 [42], ResNet in 2015 [43], DenseNet in 2016 [44], EfficientNet in 2019 [45], etc.
<br/>
Transformer is a groundbreaking deep learning architecture proposed by Google in the 2017 paper “Attention is all you need” [46]. Since then, it has rapidly surpassed RNN such as LSTM [47]/GRU [48] to become the SOTA model in Natural Language Processing (NLP). Almost all of the current Large Language Models (LLMs) are based on the transformer architecture, including Generative Pretrained Transformers (GPT) [49] [50] developed by OpenAI, Bidirectional Encoder Representations from Transformers (BERT) [51] by Google, Large Language Model Meta AI (Llama) [52] by Meta, etc. Transformers have also found wide applications besides NLP, including playing chess [53] and go [54], multi-modal processing [55], computer vision [56] [57], and many more. Specifically, a vision transformer (ViT) model was proposed in 2021 [57] and achieved excellent performance at image classification compared to CNN.
The images of 2-D IM lattices are well-qualified candidates for CNN and ViT, which are explored in this study. The architecture of CNN and ViT models will be described in Section 4.
<br/><br/>

## 1.3	Arctic sea ice
<br/>
The reversible phase transition between water and ice makes the IM a great tool to study the dynamics of a surface region with the co-existence of both states. In this study, we apply a 2-D IM lattice to study the dynamics of Arctic sea ice melting and freezing cycles, a major climate change indicator that is of significant environmental, economic and social significance [58].
<br/>
Sea ice is undoubtedly an integral part of the Arctic Ocean and the earth. In the dark winter months, ice covers almost the entirety of the Arctic Ocean, and the ice extent—defined as the percentage of the areas that are covered by at least 15% ice—and the ice thickness typically reaches its peak around March. Starting in late spring, ice melting gradually exceeds water freezing due to higher temperatures and longer hours of sunlight exposure. Sea ice typically reaches the minimum extent and thickness in mid-September, when ice coverage can drop to under half of the winter maximum [34]. After mid-September, sea water freezing starts to exceed ice melting, so ice coverage expands. This cycle repeats annually.
<br/>

Albedo, the percentage of incident light reflected from the surface of the earth, is highly dependent on the ice extent [59]. Light-colored ice or snow reflects more light than blue-colored liquid water; therefore, ice is essential to keeping the Arctic at a cooler temperature and subsequently maintaining the energy balance around the globe. If the energy balance is broken, as ice decline has been detected in recent years, the ice-albedo feedback loop effect may occur, i.e., less reflection and more absorption of solar energy, leading to even more ice loss and further global warming. Moreover, the Arctic ecosystem is inversely impacted by the decline in sea ice coverage, which, for instance, threatens the lives of polar bears and walruses who rely on sea ice for hunting and breeding [60]. 
<br/>

Data recorded by the National Aeronautics and Space Administration (NASA) and the National Snow and Ice Data Center (NSIDC) since 1979 has shown substantial declines in both ice extent and thickness in the Arctic, despite the year-over-year fluctuations in either direction. The lowest Arctic sea ice extent was observed in September of 2012 [1] [61]; between 2013 and 2022, the ice extent was higher than the 2012 minimum, but still much lower than the average of the past four decades. 2023 has recorded the hottest year by a significant margin so far [2] [62], and 2024 might break the record again [63]. Some questions then come to us naturally: how does the Artic sea ice extent in the most recent years compare to the 2012 level? Can our model simulations closely match the observations in the real data? These questions will be addressed in Section 5 and 6.
<br/><br/>

# 2.	Theoretical framework
<br/>
<h2> 2.1	Classical Ising model </h2>
<br/>
The system described by an IM is a set of lattice sites, each having a spin that interacts with its neighbors. The Hamiltonian function [4] for the lattice σ in a standard IM is given as 
<br/>

<figure>
    <img src="/images/eq1.png" >
</figure>

<!---
![image](https://github.com/user-attachments/assets/dee81799-e014-4984-adeb-0a8a3df9398e)
<br/>
<figure>
    <img src="https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/b9cbaf41-2590-46f0-9473-45629398363d)" width="250" height="50">
</figure>
-->

where σ_i represents the spin variables at site i and takes the value of +1 or -1; J_ij represents the interaction between sites i and j and can take positive values for ferromagnetic and paramagnetic materials, or negative for antiferromagnetic materials; B_i captures the interaction between the external field and site i. i and j range across the full lattice, which can be one, two or higher dimensions, and <i, j> represents pairs of spins at sites i and j that interact with each other. In a simple setup, each spin may only interact with its nearest neighbors, so <i, j> sums over adjacent sites only. For example, in a simple 2-D IM, each spin interacts only with the sites positioned immediately left, right, above, and below. 
<br/>
In statistical physics, the configuration probability of lattice σ follows the Boltzmann distribution 
<br/>
<figure>
    <img src="/images/eq2.png" >
</figure>

<!--- 
<br/>
  ![image](https://github.com/user-attachments/assets/5f266ac7-cf2c-4816-80d6-300fd73bac70)
<br/>

<figure>
    <img src="https://github.com/user-attachments/assets/5f266ac7-cf2c-4816-80d6-300fd73bac70" width="750" height="50">
</figure>
<br/>
-->

where Z is the partition function: <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/bde41b65-a38b-4ad6-a56f-61f525a37bf4)

and <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/86e2baa8-fcec-4fbb-97d2-96a360444b21)
<br/>
β is the inverse temperature; 
$k_B$ is the Boltzmann constant; T is the IM temperature, which differentiates from the ambient temperature discussed later.
<br/>
The evolution of the kinetic IM runs through a series of spin flips over the lattice. The probability of each spin flip depends on whether such a flip increases or reduces the Hamiltonian of the system. Mathematically the probability is determined by $min⁡(1,e^{(-β(H_ν-H_μ ) )})$ [64], where Hv and Hµ represent the Hamiltonian of the system before and after the flip. It can be easily seen that higher IM temperatures lead to greater thermal fluctuations and larger variances in the spin value distribution, while lower IM temperatures result in fewer fluctuations.

<br/>
<h2> 2.2 Continuous spin Ising model </h2>
<br/>
Most studies of the IM focus on binary values of the spins, i.e., σ_i takes values of +1 or -1 only. However, the sea ice data for each location takes varying values between 0 and 1 that represent the percentage of ice coverage. Therefore, we generalize the IM to allow for continuous spin values that can take any real number between -1 and +1. This generalization enables the IM to examine more realistic systems, but also adds a high degree of complexity to the mathematical solutions. Past research has studied phase transitions and critical behaviors of the continuous IM [65] [66], and recently, an IM with variable power-law spin strengths is studied with its rich phase diagrams [67].
<br/>

The Hamiltonian function of the continuous spin IM is represented by the same Equation (1). However, σ_i now takes continuous values between +1 and -1; $-J_{ij} σ_i σ_j$ reaches the minimum energy state if σ_i=σ_j=+1, or σ_i=σ_j=-1, as the energy of any other value pair is higher. The highest energy is observed when σ_i=+1, σ_j=-1, or vice versa. This numeric feature works ideally for an ice/water lattice: the most stable low energy state is either 100% water or ice across two adjacent locations, whereas full ice next to full water displays the most unstable high energy state.
<br/><br/>

<h2> 2.3 Monte Carlo simulation and inertia factor </h2>
<br/>
The incorporation of the continuous spins also adds to the complexity of the Monte Carlo (MC) simulation of the IM lattice. In the classical binary spin IM, σ_i can only flip to -σ_i in each simulation step, and therefore the absolute value of the change is always 2 no matter if the flip goes from -1 to +1 or from +1 to -1. In a continuous spin IM, the challenge of determining the post-flip numeric value of the new spin arises. In our approach, this new spin value is implemented through a random number $σ'_i$ uniformly distributed between -1 and +1, which will be explained in greater details in Section 4.4. Moreover, we incorporate an innovative inertia factor I, and the probability of each flip is determined by   <br/>
![image](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/d2ce424e-7772-4a88-9584-ca15272a8c84)
<br/>
where σ_i represents the original spin value before the change, $σ'_i$ the new attempted value, and H_ν and H_μ the system Hamiltonian before and after as described in Equation (1) and Section 2.1. 
<br/>
The newly added $-I|σ'_i-σ_i |$  accounts for the energy needed to overcome the inertia of the spin change, and I is an IM parameter to be fitted. Intuitively, this term represents the natural resistance to any state change and can also be thought of as an analog to the latent heat needed for the ice/water phase transition in classical thermodynamics. Motivated by the fact that the total energy change for water/ice phase transition at constant temperature and pressure is proportional to mass, we choose a linear functional form for the inertia term as the simplest and most sensible assumption. Therefore, the total energy required for a spin flip is $∆E=H_ν-H_μ+I|σ'_i-σ_i |$, which consists of two parts: the system Hamiltonian change plus the inertia term. The probability of spin value change follows the Boltzmann distribution as Equation (5). 
<br/>
Here is an example to illustrate the inertia effect. Starting with an initial spin value of 0.8, a flip to either 0.7 or 0.6 may result in the same system Hamiltonian value for the new lattice. However, we differentiate these two new states by assigning higher probability for the flip to 0.7 because the spin change is smaller. In Equation (5), $-I|σ'_i-σ_i |$ influences the distribution of new spin values, and in practice, it significantly improves the simulation results to better match the observations.
<br/>
In summary, we introduce to the classical IM the continuous spin values and a novel inertia factor. These mathematical additions prepare us to study real-world Arctic sea ice dynamics while keeping the computational complexity tractable.
<br/><br/>

<h2> 2.4 The inverse Ising problem: solved with deep neural networks </h2>
<br/>
There has been various machine learning research on the IM, many of which employes CNN due to the tremendous power of CNN on image recognition. These studies focus on exploring the phase transitions near a critical temperature [68], while some of them involve generative neural networks such as variational autoencoders [69] or normalizing flows [70]. My task in this study is different, which is to solve the so-called inverse Ising problem [71]: given the start and end state images of the Ising lattices, how do we determine the IM interaction parameters (J, B, I)? In this paper, we will train a few different deep learning models including CNN and ViT for this task, with detailed steps to be explained in Section 4.5.
<br/>
The key to CNN is convolutional layers, which employ a mathematical operation called convolution. A convolutional layer consists of kernels, or filters. The kernels slide along the input grid and compute the weighted sums, as shown below:
<br/>
![image](https://github.com/user-attachments/assets/e1f0b6f4-9ca7-4163-bd22-6e46a1b59c09)
<br/>
Where i is the two-dimensional image; K represents the kernel; the convolution operator is typically denoted as an asterisk *. 
<br/>
In most CNNs, the convolutional layers are followed by pooling layers, which reduce the network size and generate a summary statistic from the outputs of the convolutional layers. For instance, max pooling is one of the most popularly used techniques, which calculates the maximum value within a neighboring rectangular area. 
<br/>
AlexNet [40], a CNN network comprising 5 convolutional layers, demonstrated that the depth of neural networks were essential to their performance by winning the ImageNet Large Scale Visual Recognition Challenge in 2012 [72]. Since then, deeper networks gained popularity as they outperform the shallower ones [73]. However, deeper networks are more difficult to train due to vanishing/exploding gradients [74] [75] and the degradation [76] problems, which were overcome by the breakthrough of the residual network in 2015 [43]. Specifically, for a subnetwork with input x and the underlying network function H(x), instead of directly learning H(x), the corresponding residual network learns a new function F(x) defined as:
<br/>
![image](https://github.com/user-attachments/assets/2a36a7d4-e0f5-4c8f-ab8c-882aa8d2ca26)
<br/>
F(x), called residual function, is implemented as short skip connections. ResNet [43], a residual network as deep as over 100 layers, achieved superior performance in  image classification than any previous models.
<br/>
Vision transformer (ViT) [57] was developed as alternatives to CNN in computer vision tasks. The core of the transformer architecture is the self-attention mechanism. Long range dependencies and relationships between the inputs, either a sequence of texts in NLP or image patches in ViT, are captured via scaled dot-product attention [46] as illustrated below, which is one of the most influential formulas in deep learning:
<br/>
![image](https://github.com/user-attachments/assets/82af217f-0304-461f-b987-716b6d23998b)
<br/>
Where Q represents the query matrix, K the key matrix, V the value matrix; K^Tis the transpose of K; QK^Tis the matrix multiplication; d_k is the dimension of the keys. Softmax function for any vector x=(x1,x2,…,xn) is defined as:
<br/>
![image](https://github.com/user-attachments/assets/acf62a19-2bcd-4493-b3bb-3dccdd97ec07)
<br/>
The weights of Q, K, V are trained to learn the relationship between different parts of the inputs; the transformer outputs can be fed to various downstream task, e.g. a multi-layer perceptron (MLP) [77] for image classification.
<br/>
In this study, we will build three neural networks—a simple CNN from scratch, a much deeper fine-tuned ResNet, and a ViT—and apply each of them to solve the inverse Ising problem independently.
<br/><br/>

# 3. Data description
<br/>

Our study uses the “Near-Real-Time DMSP SSMIS Daily Polar Gridded Sea Ice Concentrations” (NRTSI) dataset [78] collected by the National Snow and Ice Data Center (NSIDC). It captures daily sea ice concentrations for both the Northen and Southern Hemispheres. The Special Sensor Microwave Imager/Sounder (SSMIS) on the NANA Defense Meteorological Satellite Program (DMSP) satellites acquires the near-real-time passive microwave brightness temperatures, which serve as inputs to the NRTSI dataset using the NASA Team algorithm to generate the sea ice concentrations.
<br/>
The NRTSI files are in netCDF format. Each file of the Arctic region contains a lattice of 448 rows by 304 columns, covering a large earth surface area with the north pole at the center. Each grid cell represents an area of approximately 25 kilometers by 25 kilometers. The value for each grid cell is an integer from 0 to 250 that indicates the fractional ice coverage scaled by 250. 0 indicates 0% ice concentration; 250 indicates 100% ice concentration. The image of part of the NRTSI file on Sept 16th, 2022 is illustrated in Figure 1(a). In the map, white represents ice, blue represents water, and gray represents land. The exact north pole location is covered by a gray circular mask because of the limitation of the satellite sensor measurement caused by the orbit inclination and instrument swath.
<br/>

<!---
!["Figure 1: Arctic sea ice and the focus area"](/images/Figure1.jpg)
-->

<figure>
    <img src="/images/Figure1.jpg" width="400" height="250">
    <figcaption> Figure 1: (a) Arctic sea ice NRTSI image on Sept 16, 2022 and (b) the focus area of 1,500km X 1,500km </figcaption>
</figure>

<br/><br/>
For this research paper, we focus on studying a specific geographic region bounded by the black square in Figure 1(a), ranging from the East Siberian Sea (to the top of the box) and the Beaufort Sea (to the left of the box) to near the polar point, and the red oval marks the Canadian Arctic Archipelago area to be discussed later. A zoom-in image of this focus area is shown in Figure 1(b) . This large square area is unobstructed by land or the north pole mask, making it an ideal field for the IM lattice setup. The area contains 60 rows and 60 columns in the data file, covering approximately 1500km x 1500km, or about 2.25 million square kilometers.

<br/><br/>

# 4.    Ising model and neural network setup
<br/>
The methodology of our study on sea ice dynamics is outlined as follows: we first normalize the NRSTI data to a continuous Ising lattice, carefully choose the simulation periods, and set up the Ising parameters (J, B, I) to be fitted. Then given the initial lattice of each simulation period, we run the Metropolis MC simulation based on the values of (J, B, I) to generate a final state of the Ising lattice for this period. The full Metropolis simulation procedure is passed into a neural network to solve the inverse Ising problem, i.e., to find the Ising parameters after training so that the simulated final Ising lattice matches the observed NRSTI data as closely as possible.
<br/>
<h2> 4.1	Ising model lattice </h2>
<br/>
We first transform the NRTSI data of the focus region as shown in Figure 1 (b) to Ising-style data. A simple linear mapping is applied to convert integers from 0 to 250 to real numbers from -1 to +1. -1 indicates the cell is 100% ice; +1 indicates 100% water; 0 indicates 50%/50% coverage of water/ice. Each cell covers 25km x 25km of the total 1500km x 1500km focus region, and therefore a 60x60 matrix is initialized as the 2-D IM lattice for our study. 
<br/><br/>

<h2> 4.2	Simulation periods </h2>
<br/>
Figure 2 (a) and (b) display an example of the initial and the final target states of an IM lattice simulation run. The simulation periods are chosen to be consistently half a month apart, for example, Sept 16th, 2022 in Figure 2 (a) and Oct 1st, 2022 in Figure 2 (b). This semi-monthly frequency is chosen to balance two considerations. First, the period is sufficiently long to allow for sizable differentiation in the ice/water configurations between the start and the end dates; second, the period is not excessively too long and allows the IM simulation to mimic the daily water/ice evolution on the interim dates between the start and the end, which is to be illustrated in Section 5.3. 
<br/>

<figure>
    <img src="/images/Figure2.png" width="480" height="250">
    <figcaption> Figure 2: The initial and the final target states of an IM lattice simulation run. (a) shows the actual configuration observed in the focus area on Sept 16th, 2022 and (b) on Oct 1st, 2022. Each full simulation period is half a month. Blue color indicates water; white indicates ice. The darker the color on each cell, the higher the water concentration, as shown by the scale on the right. </figcaption>
</figure>

<br/><br/>
<h2> 4.3	Ising model parameters </h2>
<br/>
In the IM Hamiltonian function, i.e., Equation (1), We set the following:
<br/>
● σ_i is a real number between -1 and +1 for any cell i in the focus area.
<br/>
● <i, j> sums over all adjacent cells, so each spin interacts only with four sites that are positioned immediately left, right, above and below.
<br/>
● J_ij  is set to be constant within each simulation period across all cells.
<br/>
● Bi is set to be time-invariant within each simulation period. However, in order to capture the real-world external force variation across locations, especially the environmental differences from the coast area to the north pole, Bi is set to be a linear function of x_i (the row) and y_i (the column) coordinates of cell i, i.e., Bi = B_0+B_x (x_i-x_0 )+B_y (y_i-y_0), where B_0 is the average B over the lattice, and x_0 and y_0 are the coordinates of the lattice center.
<br/>
● I, the inertia factor, is set to be constant within each simulation period.
<br/>
● β, the inverse Boltzmann temperature, is set to 1 without loss of generality.
<br/><br/>

<h2> 4.4	Metropolis simulation setps </h2>
<br/>
Various Monte Carlo (MC) methods have been developed for the IM simulation. Among them the most widely used are the Glauber dynamics and the Metropolis-Hasting algorithm. In this study, we follow the latter for the MC simulation of the IM lattice evolution. As described in Section 2.3, an inertia factor is introduced into our model and the generalized Metropolis-Hastings MC steps are below:
<br/>
1. Select cell i at random from the 2-D lattice of the focus area. Let the spin value of this cell be σ_i.
<br/>
2. Generate another uniform random variable 〖σ^'〗_i between -1 and +1.
<br/>
3. Compute the energy change ∆Hi= H_ν-H_μ  from σ_i to 〖σ^'〗_i. 
<br/>
4. Compute the energy I|〖σ^'〗_i-σ_i | to overcome the inertia of changing the spin value at i.
<br/>
5. Compute the total energy change ∆E = ∆Hi  +I|〖σ^'〗_i-σ_i |.  
<br/>
6. (a) If ∆E is negative, the energy change is favorable since the energy is reduced. The spin value change is therefore accepted to 〖σ^'〗_i.
<br/>
   (b) If ∆E is positive, the probability of the spin flip is determined by the Boltzmann distribution. In this case, another uniform random variable r between 0 and 1 is generated. If r is less than P = e^(-β∆E), the spin value change is accepted; otherwise, the change is rejected and the spin value at i stays at σ_i.
<br/><br/>

For each semi-monthly simulation period, we repeat the above MC steps 50,000 times. As the lattice of our focus area has 3,600 cells, this repetition allows approximately 14 flip tries for each cell, or roughly once per day. This specific repetition number is chosen by taking into account the computational complexity of the algorithm and also making sure that each cell of the Ising lattice gets sufficient attempts to be changed. Other choices of the repetition number can be considered, which may result in different fitted parameter values. What is important is to ensure the number of repetitions for each period proportional to its duration, so the time unit of each Metropolis step is the same across the full simulation process [79].
<br/><br/>

<h2> 4.5 Architecture of the neural networks </h2>
<br/>
In this research, we apply deep neural network models to solve the inverse Ising problem; that is, to find the best-fit Ising parameters (J, B_0,〖 B〗_x 〖,B〗_y, I) based on the initial and final states of each simulation period. Three models are implemented: a simple CNN built from scratch, a much deeper fine-tuned ResNet and ViT respectively.
<br/>
The architecture of my first simple CNN is illustrated in Figure 3, which is similar to AlexNet [40]. It starts with the input layer, which consists of two images of shape (60, 60, 2), representing the start and end state images respectively. It is followed by four convolutional layers with a kernel shape (3, 3). The kernel counts from 16 in the first convolutional layer to 32, 64, and 128 in the last layer. Zero padding and strides (1, 1) are used to ensure coverage of the entire input grid. Each of the convolutional layers applies a Leaky Rectified Linear Unit (LeakyReLU) activation function.  Every convolutional layer is followed by a max pooling layer of pool size (2, 2) that summarizes the crucial features and reduces the layer size. The outputs of the last max pooling layer are flattened and followed by a fully connected dense layer and a dropout layer to avoid overfitting. The outputs are fed to the final dense layer with 5 neurons. It is worth noting that our CNN model differs from most of the CNNs used for classification tasks, as our targets are the continuous Ising parameters instead of discrete categorical labels. Therefore, a linear activation function is chosen for the final layer, rather than other popular choices such as Sigmoid in classification tasks.  The total number of trainable parameters stays at 213,101, making this a small deep learning algorithm that can be trained very fast on an Intel i7-11700F CPU. This CNN model is implemented with the Tensorflow/Keras [80] package in IIMCNNModel.py.
<br/>

<figure>
    <img src="/images/Figure3.png" width="650" height="400">
    <figcaption> Figure 3: Architecture diagram of the simple CNN model to solve the inverse Ising problem </figcaption>
</figure>
<br/><br/>

Our second network is a much deeper ResNet with weights pretrained on the large ImageNet dataset [81]. The original paper by He et al. [43] developed 5 variations with different network depths: ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152. My study employs ResNet50, as a balance choice between the size and performance of the network. In summary, ResNet50 consists of 49 trainable convolutional layers and 1 fully connected layer at the end. In this research, the model is tailored to receive inputs of image shape 60x60 and 2 channels, which then passes the ResNet50 network; finally, a fully connected linear layer with 5 output neurons is appended at the end to learn the 5 Ising parameters (J, B_0,〖 B〗_x 〖,B〗_y, I). The high-level architecture of this fine-tuned network is illustrated in Figure 4(a).  Total number of trainable parameters of this network is 25,558,901, and it can be trained on a Nvidia GeForce RTX3060 GPU in approximately 35 minutes. This fine-tuned model is implemented under the PyTorch [82] framework using the built-in TorchVision ResNet package [83] as in IIMResnetFineTune.py.
<br/>
The last network in this study is a fine-tuned ViT with weights also pretrained on the ImageNet dataset. Since the original paper by Dosovitskiy et al. [57], various ViT implementations have been developed, including Data-efficient Image Transformers (DeiT) [84] by Meta, BERT Pre-training of Image Transformers (BEiT) [85] by Microsoft, etc. In this research, we fine-tune the pretrained google/vit-base-patch16-224 model [86], available in the Transformer package [87] as implemented by Hugging Face [88], a collaboration platform which warehouses a collection of open-source machine learning models. as illustrated in Figure 4(b), this base ViT network consists of 12 sequential transformer encoder blocks, each of which consists of a layer-norm (LN), a multi-head self-attention network, a multi-layer perceptron with Gaussian Error Linear Unit (GELU) activation, and residual connections. In this research, the model is customized for inputs of patches of 60x60 images with 2 channels, and the final output is converted to a 5-neuron fully connected linear layer. The total number of trainable parameters is 85,259,525; the network is more compute-heavy due to the quadratic complexity when calculating the attention matrices. It takes about 70 hours to train this transformer model on an RTX3060 GPU. This model is implemented in IIMViTFineTune.py.
<br/>
<figure>
    <img src="/images/Figure4.png" width="500" height="500">
    <figcaption> Figure 4: (a) Architecture of the customized (a) ResNet50, and (b) ViT networks used in this research. Bulk of the architecture diagrams are taken from He et al. [43] and Dosovitskiy et al. [57] </figcaption>
</figure>
<br/><br/>
<h2> 4.6 Training data for the neural networks </h2>
<br/>
Training neural networks requires a substantial amount of data. In my study, these data are generated following the simulation steps described in previous subsections. To be specific, we start with the Ising lattice at the initial state of a simulation period and randomly select 10,000 set of parameters (J, B_0,〖 B〗_x 〖,B〗_y, I); for each set of parameters, we run the Metropolis simulation steps as described in section 4.4. As a result, we generate 10,000 sets of training samples corresponding to each of the initial states. An example of the training sample corresponding to the initial state of the focus area on Sept 16th, 2022 and Ising parameters (J = 2.31, B_0=-14.5,〖 B〗_x=-6.15〖,B〗_y=0.07, I = 9.93) is illustrated in Figure 5. Compared with Figure 2, this training sample apparently happens to correspond to a much faster freezing cycle than the actual observation.
<br/>
<figure>
    <img src="/images/Figure5.png" width="400" height="240">
    <figcaption> Figure 5: A training sample pair. (a) is the initial observed state on Sept 16th, 2022 and (b) the final simulated state on Oct 1st, 2022 based on Ising parameters (J=2.31, B0=-14.5, Bx=-6.15, By=0.07, I=9.93). </figcaption>
</figure>
<br/><br/>
These generated Ising configuration pairs for all simulation periods from June 16th to Jan 1st are passed as the inputs to our neural network. As a supervised learning process, the target of our network is set to be the corresponding Ising parameters. After the network is fully trained, estimating the best-fit Ising parameters for each of our simulation periods is straightforward: we simply pass the observed initial and end state sea ice images to the network, which predicts and returns the respective Ising parameters.
<br/><br/><br/>








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


