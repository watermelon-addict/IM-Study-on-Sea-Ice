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

where $σ_i$ represents the spin variables at site i and takes the value of +1 or -1; $J_{ij}$ represents the interaction between sites i and j and can take positive values for ferromagnetic and paramagnetic materials, or negative for antiferromagnetic materials; B_i captures the interaction between the external field and site i. i and j range across the full lattice, which can be one, two or higher dimensions, and <i, j> represents pairs of spins at sites i and j that interact with each other. In a simple setup, each spin may only interact with its nearest neighbors, so <i, j> sums over adjacent sites only. For example, in a simple 2-D IM, each spin interacts only with the sites positioned immediately left, right, above, and below. 
<br/>
In statistical physics, the configuration probability of lattice σ follows the Boltzmann distribution 
<br/>
<figure>
    <img src="/images/eq2.png" >
</figure>

where Z is the partition function: 
<br/>
<figure>
    <img src="/images/eq3.png" >
</figure>

and 
<br/>
<figure>
    <img src="/images/eq4.png" >
</figure>

β is the inverse temperature; 
$k_B$ is the Boltzmann constant; T is the IM temperature, which differentiates from the ambient temperature discussed later.
<br/>
The evolution of the kinetic IM runs through a series of spin flips over the lattice. The probability of each spin flip depends on whether such a flip increases or reduces the Hamiltonian of the system. Mathematically the probability is determined by $min⁡(1,e^{(-β(H_ν-H_μ ) )})$ [64], where $H_v$ and $H_µ$ represent the Hamiltonian of the system before and after the flip. It can be easily seen that higher IM temperatures lead to greater thermal fluctuations and larger variances in the spin value distribution, while lower IM temperatures result in fewer fluctuations.

<br/>
<h2> 2.2 Continuous spin Ising model </h2>
<br/>
Most studies of the IM focus on binary values of the spins, i.e., σ_i takes values of +1 or -1 only. However, the sea ice data for each location takes varying values between 0 and 1 that represent the percentage of ice coverage. Therefore, we generalize the IM to allow for continuous spin values that can take any real number between -1 and +1. This generalization enables the IM to examine more realistic systems, but also adds a high degree of complexity to the mathematical solutions. Past research has studied phase transitions and critical behaviors of the continuous IM [65] [66], and recently, an IM with variable power-law spin strengths is studied with its rich phase diagrams [67].
<br/>

The Hamiltonian function of the continuous spin IM is represented by the same Equation (1). However, $σ_i$ now takes continuous values between +1 and -1; $-J_{ij} σ_i σ_j$ reaches the minimum energy state if $σ_i=σ_j=+1$, or $σ_i=σ_j=-1$, as the energy of any other value pair is higher. The highest energy is observed when $σ_i=+1$, $σ_j=-1$, or vice versa. This numeric feature works ideally for an ice/water lattice: the most stable low energy state is either 100% water or ice across two adjacent locations, whereas full ice next to full water displays the most unstable high energy state.
<br/>
<br/>

## 2.3 Monte Carlo simulation and inertia factor
<br/>

The incorporation of the continuous spins also adds to the complexity of the Monte Carlo (MC) simulation of the IM lattice. In the classical binary spin IM, $σ_i$ can only flip to $-σ_i$ in each simulation step, and therefore the absolute value of the change is always 2 no matter if the flip goes from -1 to +1 or from +1 to -1. In a continuous spin IM, the challenge of determining the post-flip numeric value of the new spin arises. In our approach, this new spin value is implemented through a random number $σ'_i$ uniformly distributed between -1 and +1, which will be explained in greater details in Section 4.4. Moreover, we incorporate an innovative inertia factor I, and the probability of each flip is determined by   
<br/>
<figure>
    <img src="/images/eq5.png" >
</figure>

where σ_i represents the original spin value before the change, $σ'_i$ the new attempted value, and $H_ν$ and $H_μ$ the system Hamiltonian before and after as described in Equation (1) and Section 2.1. 
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
<figure>
    <img src="/images/eq6.png" >
</figure>

Where i is the two-dimensional image; K represents the kernel; the convolution operator is typically denoted as an asterisk *. 
<br/>
In most CNNs, the convolutional layers are followed by pooling layers, which reduce the network size and generate a summary statistic from the outputs of the convolutional layers. For instance, max pooling is one of the most popularly used techniques, which calculates the maximum value within a neighboring rectangular area. 
<br/>
AlexNet [40], a CNN network comprising 5 convolutional layers, demonstrated that the depth of neural networks were essential to their performance by winning the ImageNet Large Scale Visual Recognition Challenge in 2012 [72]. Since then, deeper networks gained popularity as they outperform the shallower ones [73]. However, deeper networks are more difficult to train due to vanishing/exploding gradients [74] [75] and the degradation [76] problems, which were overcome by the breakthrough of the residual network in 2015 [43]. Specifically, for a subnetwork with input x and the underlying network function $H(x)$, instead of directly learning $H(x)$, the corresponding residual network learns a new function $F(x)$ defined as:
<br/>
<figure>
    <img src="/images/eq7.png" >
</figure>
<br/>
F(x), called residual function, is implemented as short skip connections. ResNet [43], a residual network as deep as over 100 layers, achieved superior performance in  image classification than any previous models.
<br/>
Vision transformer (ViT) [57] was developed as alternatives to CNN in computer vision tasks. The core of the transformer architecture is the self-attention mechanism. Long range dependencies and relationships between the inputs, either a sequence of texts in NLP or image patches in ViT, are captured via scaled dot-product attention [46] as illustrated below, which is one of the most influential formulas in deep learning:
<br/>
<figure>
    <img src="/images/eq8.png" >
</figure>
<br/>

Where Q represents the query matrix, K the key matrix, V the value matrix; $K^T$ is the transpose of K; $QK^T$ is the matrix multiplication; $d_k$ is the dimension of the keys. Softmax function for any vector $x=(x_1,x_2,…,x_n)$ is defined as:
<br/>
<figure>
    <img src="/images/eq9.png" >
</figure>


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
The methodology of our study on sea ice dynamics is outlined as follows: we first normalize the NRSTI data to a continuous Ising lattice, carefully choose the simulation periods, and set up the Ising parameters $(J, B, I)$ to be fitted. Then given the initial lattice of each simulation period, we run the Metropolis MC simulation based on the values of $(J, B, I)$ to generate a final state of the Ising lattice for this period. The full Metropolis simulation procedure is passed into a neural network to solve the inverse Ising problem, i.e., to find the Ising parameters after training so that the simulated final Ising lattice matches the observed NRSTI data as closely as possible.
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
## 4.3	Ising model parameters
<br/>
In the IM Hamiltonian function, i.e., Equation (1), We set the following:
<br/><br/>

●  $σ_i$ is a real number between -1 and +1 for any cell i in the focus area.
<br/>
●  <i, j> sums over all adjacent cells, so each spin interacts only with four sites that are positioned immediately left, right, above and below.
<br/>
●  $J_{ij}$  is set to be constant within each simulation period across all cells.
<br/>
●  $B_i$ is set to be time-invariant within each simulation period. However, in order to capture the real-world external force variation across locations, especially the environmental differences from the coast area to the north pole, $B_i$ is set to be a linear function of $x_i$ (the row) and $y_i$ (the column) coordinates of cell i, i.e., $B_i = B_0+B_x (x_i-x_0 )+B_y (y_i-y_0)$, where $B_0$ is the average B over the lattice, and $x_0$ and $y_0$ are the coordinates of the lattice center.
<br/>
●  I, the inertia factor, is set to be constant within each simulation period.
<br/>
●  β, the inverse Boltzmann temperature, is set to 1 without loss of generality.
<br/><br/>

## 4.4	Metropolis simulation setups
<br/>
Various Monte Carlo (MC) methods have been developed for the IM simulation. Among them the most widely used are the Glauber dynamics and the Metropolis-Hasting algorithm. In this study, we follow the latter for the MC simulation of the IM lattice evolution. As described in Section 2.3, an inertia factor is introduced into our model and the generalized Metropolis-Hastings MC steps are below:
<br/><br/>
1.  Select cell i at random from the 2-D lattice of the focus area. Let the spin value of this cell be $σ_i$.
<br/>
2.  Generate another uniform random variable $σ'_i$ between -1 and +1.
<br/>
3.  Compute the energy change $∆H_i= H_ν-H_μ$  from $σ_i$ to $σ'_i$. 
<br/>
4.  Compute the energy $I|σ'_i-σ_i|$ to overcome the inertia of changing the spin value at i.
<br/>
5.  Compute the total energy change $∆E = ∆H_i  + I|σ'_i-σ_i|$.  
<br/>
6.  (a) If ∆E is negative, the energy change is favorable since the energy is reduced. The spin value change is therefore accepted to $σ'_i$.
<br/>
   (b) If ∆E is positive, the probability of the spin flip is determined by the Boltzmann distribution. In this case, another uniform random variable r between 0 and 1 is generated. If r is less than $P = e^(-β∆E)$, the spin value change is accepted; otherwise, the change is rejected and the spin value at i stays at $σ_i$.
<br/><br/>

For each semi-monthly simulation period, we repeat the above MC steps 50,000 times. As the lattice of our focus area has 3,600 cells, this repetition allows approximately 14 flip tries for each cell, or roughly once per day. This specific repetition number is chosen by taking into account the computational complexity of the algorithm and also making sure that each cell of the Ising lattice gets sufficient attempts to be changed. Other choices of the repetition number can be considered, which may result in different fitted parameter values. What is important is to ensure the number of repetitions for each period proportional to its duration, so the time unit of each Metropolis step is the same across the full simulation process [79].
<br/><br/>

## 4.5 Architecture of the neural networks
<br/>
In this research, we apply deep neural network models to solve the inverse Ising problem; that is, to find the best-fit Ising parameters $(J, B_0, B_x, B_y, I)$ based on the initial and final states of each simulation period. Three models are implemented: a simple CNN built from scratch, a much deeper fine-tuned ResNet and ViT respectively.
<br/>
The architecture of my first simple CNN is illustrated in Figure 3, which is similar to AlexNet [40]. It starts with the input layer, which consists of two images of shape (60, 60, 2), representing the start and end state images respectively. It is followed by four convolutional layers with a kernel shape (3, 3). The kernel counts from 16 in the first convolutional layer to 32, 64, and 128 in the last layer. Zero padding and strides (1, 1) are used to ensure coverage of the entire input grid. Each of the convolutional layers applies a Leaky Rectified Linear Unit (LeakyReLU) activation function.  Every convolutional layer is followed by a max pooling layer of pool size (2, 2) that summarizes the crucial features and reduces the layer size. The outputs of the last max pooling layer are flattened and followed by a fully connected dense layer and a dropout layer to avoid overfitting. The outputs are fed to the final dense layer with 5 neurons. It is worth noting that our CNN model differs from most of the CNNs used for classification tasks, as our targets are the continuous Ising parameters instead of discrete categorical labels. Therefore, a linear activation function is chosen for the final layer, rather than other popular choices such as Sigmoid in classification tasks.  The total number of trainable parameters stays at 213,101, making this a small deep learning algorithm that can be trained very fast on an Intel i7-11700F CPU. This CNN model is implemented with the Tensorflow/Keras [80] package in IIMCNNModel.py.
<br/>

<figure>
    <img src="/images/Figure3.png" width="650" height="400">
    <figcaption> Figure 3: Architecture diagram of the simple CNN model to solve the inverse Ising problem </figcaption>
</figure>
<br/><br/>

Our second network is a much deeper ResNet with weights pretrained on the large ImageNet dataset [81]. The original paper by He et al. [43] developed 5 variations with different network depths: ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152. My study employs ResNet50, as a balance choice between the size and performance of the network. In summary, ResNet50 consists of 49 trainable convolutional layers and 1 fully connected layer at the end. In this research, the model is tailored to receive inputs of image shape 60x60 and 2 channels, which then passes the ResNet50 network; finally, a fully connected linear layer with 5 output neurons is appended at the end to learn the 5 Ising parameters $(J, B_0, B_x, B_y, I)$. The high-level architecture of this fine-tuned network is illustrated in Figure 4(a).  Total number of trainable parameters of this network is 25,558,901, and it can be trained on a Nvidia GeForce RTX3060 GPU in approximately 35 minutes. This fine-tuned model is implemented under the PyTorch [82] framework using the built-in TorchVision ResNet package [83] as in IIMResnetFineTune.py.
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
Training neural networks requires a substantial amount of data. In my study, these data are generated following the simulation steps described in previous subsections. To be specific, we start with the Ising lattice at the initial state of a simulation period and randomly select 10,000 set of parameters $(J, B_0, B_x, B_y, I)$; for each set of parameters, we run the Metropolis simulation steps as described in section 4.4. As a result, we generate 10,000 sets of training samples corresponding to each of the initial states. An example of the training sample corresponding to the initial state of the focus area on Sept 16th, 2022 and Ising parameters $(J = 2.31, B_0=-14.5, B_x=-6.15, B_y=0.07, I = 9.93)$ is illustrated in Figure 5. Compared with Figure 2, this training sample apparently happens to correspond to a much faster freezing cycle than the actual observation.
<br/>
<figure>
    <img src="/images/Figure5.png" width="400" height="240">
    <figcaption> Figure 5: A training sample pair. (a) is the initial observed state on Sept 16th, 2022 and (b) the final simulated state on Oct 1st, 2022 based on Ising parameters $(J = 2.31, B_0=-14.5, B_x=-6.15, B_y=0.07, I = 9.93)$. </figcaption>
</figure>
<br/><br/>
These generated Ising configuration pairs for all simulation periods from June 16th to Jan 1st are passed as the inputs to our neural network. As a supervised learning process, the target of our network is set to be the corresponding Ising parameters. After the network is fully trained, estimating the best-fit Ising parameters for each of our simulation periods is straightforward: we simply pass the observed initial and end state sea ice images to the network, which predicts and returns the respective Ising parameters.
<br/><br/><br/>

# 5. Results and analyses

<br/>
Thanks to the publicly accessible NRTSI data, simulation and training can be completed with all three networks—the simple CNN, the much deeper ResNet50, and the ViT—for every year in the past four decades. The performance of these networks varies: ResNet50 demonstrated a slight advantage in terms of both alignment with the actual data and the balance of the computational resources required for model training. Due to space constraints, in this section we present only the results from ResNet50. A comparative analysis of the three networks is discussed in Section 6.2, and detailed results with the other two networks can be found in Appendix A.1.
<br/>
<h2> 5.1  Simulation results for 2023 </h2>
<br/>
For illustration purposes, we focus on the results for 2023, the year with the most recent full annual data and critically setting the hottest year record on the earth. Results for certain other years can be found in Appendix A.2.
<br/>
Figure 6 shows the semi-monthly NRTSI sea ice images in the focus area from June 16th, 2023 to Jan 1st, 2024. As can be seen, the melting cycle starts from June 16th and goes until Sept 16th, and the freezing cycle from Sept 16th to year end. Prior to June 16th, the region is almost fully covered by ice, so the IM simulation will be trivial. This is why we set the simulation start date on June 16th of each year. During the period of June 16th to Dec 16th, every succeeding image shows considerable ice coverage difference from the previous date while retaining certain core features. This semi-monthly frequency choice allows our IM simulation to capture the essence of the evolution dynamics without overfitting the model.
<br/>
<br/>
<figure>
    <img src="/images/Figure6.png" width="600" height="400">
    <figcaption> Figure 6: The actual semi-monthly sea ice evolution in the focus area in 2023: (a) June 16th, (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2023, and (n) Jan 1st, 2024.  </figcaption>
</figure>
<br/>
<br/>
The Ising parameters $(J, B_0, B_x, B_y, I)$ for each simulation period in 2023 predicted by the fine-tuned ResNet50 model are shown in Table 1. The spin interaction coefficient J and the inertia factor I are relatively stable; intuitively, the strength of such interactions does not change much across different time periods. Moreover, J remains positive across all periods, confirming that adjacent cells are inclined to maintain values of the same sign, i.e., the area surrounding ice will be more likely to freeze, and that surrounding water will tend to melt. In this sense, the ice/water system displays the feature of ferromagnetism/paramagnetism instead of antiferromagnetism. 
<br/>
<br/>
<figure>
    <img src="/images/Table1.png" width="1000" height="180">
    <figcaption> Table 1: ResNet50 predicted Ising parameters for the 2023 sea ice evolution</figcaption>
</figure>
<br/>
<br/>
On the other hand, the external force parameters $B_0, B_x, B_y$ display large variations across different time periods. In particular, the average force $B_0$  is positive from June 1st to Sept 16th but turns negative afterwards, which can be explained by the seasonal ambient temperature as the dominant external factor for ice/water dynamics. Ambient temperature is not the only factor, though. Arctic temperature normally peaks in July/August while $B_0$ remains positive and ice melting continues through mid-September. This lag effect could be explained by other environmental effects such as albedo or jet streams but is beyond the scope of this study. 
<br/>
All values of Bx are negative due to the geographic distribution of ice coverage. For our Ising lattice representing the focus area,  x coordinates corresponding to the rows in the lattice increase from top to bottom; y coordinates for the columns increase from left to right. Interestingly, ice coverage near the bottom of our area, the Canadian Arctic Archipelago marked by the red oval in Figure 1, is much thicker than elsewhere including the north pole (the gray circular mask). In fact, many scientists believe this region will have the last piece of ice standing in the Arctic if the Blue Ocean Event happens. As the lower part of the focus area tends to have greater ice coverage, Bx is all negative. Whereas By is less negative and shows positive for certain periods, implying that the impact of the geographic location along the y direction is less pronounced than that of x. This is because the ice at the north pole is thinner than in Archipelago, which mitigates the impact of the y coordinate.  In addition, the values of Bx and By exhibit greater fluctuations than other parameters, indicating that our simplified linear functional form of $B_i = B_0+B_x (x_i-x_0 )+B_y (y_i-y_0)$ is far from perfectly modeling the full effect of external fields; it can be further enriched by linking to actual geographical and environmental factors to enhance the power of the Ising model, which is left for our future research.
<br/>
The simulated sea ice images for each 2023 period are shown in Figure 7 utilizing the Ising parameters in Table 1. These images exhibit excellent similarity to Figure 6, demonstrating the strong explanatory power of our Ising model. Nevertheless, our model is not perfect. Upon close inspection, The images in Figure 6 and Figure 7 do reveal discrepancies, especially as shown in images (d) Aug 1st and (i) Oct 16th, where the actual ice configurations display significant irregularity compared to the prior period. While an IM with simple parameterization encounters difficulties in describing these local irregularities, it is feasible to include a richer set of parameters or to employ more complicated parametric functional forms at the potential cost of overfitting. In this paper, we keep our Ising model tractable and accept these local discrepancies.
<br/>
<br/>
<figure>
    <img src="/images/Figure7.png" width="600" height="400">
    <figcaption> Figure 7: The simulated semi-monthly sea ice evolution in the focus area in 2023. (a) is the actual image on June 16th, 2023; (b) - (n) are simulated images on (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2022, and (n) Jan 1st, 2024. </figcaption>
</figure>
<br/>
<br/>
To quantify the similarity between the IM simulated configurations and the observed images, the absolute differences in ice coverages across the entire focus area for each of the simulation period in Figure 6 and Figure 7 are calculated; the results are illustrated as the heatmaps in Figure 8(a) – (n), where light yellow color indicates that the actual and simulated images match well, whereas red patches are associated with the locations that display large discrepancy. The heatmaps are very revealing: the small red patches mostly appear around the boundaries between water and ice, implying that most of the discrepancy between the simulated and actual images happens around these border areas. This is not surprising: the IM needs improvement to perfectly model these boundary granularities, but it does have strong explanatory power to capture the overall patterns.
<br/>
<br/>
<figure>
    <img src="/images/Figure8.png" width="600" height="400">
    <figcaption> Figure 8: Heatmaps illustrating the absolute difference in ice coverages between Figure 6 and Figure 7 for each semi-monthly period, from (a) June 16th, 2023 to (n) Jan 1st, 2024.  Yellow color indicates a good match and red a large difference, as shown by the scale on the right. </figcaption>
</figure>
<br/>
<br/>
<h2> 5.2    Ice coverage percentage and ice extent </h2>
<br/>
Furthermore, we compute two key numerical measures: the ice coverage percentage, i.e., the mean of the ice coverage across the entire lattice, and the ice extent, i.e., the percentage of areas that are covered by at least 15% ice. The comparisons between the actual observations and the simulation results for our focus area are shown in Figure 9. As anticipated, we see an excellent match in both figures as a result of the superior explanatory power of our IM, although the results do show marginal but non-trivial discrepancy. We can see that the simulated ice extent drops to nearly 30% in Sept 2023, one of the lowest in recorded history.
<br/>
<br/>
<figure>
    <img src="/images/Figure9.png" width="600" height="300">
    <figcaption>Figure 9: (a) The ice coverage percentage in the focus area from June 16th, 2023 to Jan 1st, 2024; (b) The ice extent for the same periods. Blue curves are the actual measures from the NRTSI data; orange ones show the IM simulation results. </figcaption>
</figure>
<br/>
<br/>
<h2> 5.3	Daily sea ice evolution </h2>
<br/>
Do these semi-monthly IM simulation results match the actual sea ice dynamics on a shorter time scale? To answer this question, we utilize the semi-monthly Ising parameters in Table 1 to simulate the daily evolution. Two periods, a melting period from Aug 1st to Aug 16th, 2023, and a freezing period from Oct 16th to Nov 1st, 2023, are simulated day-by-day for this experiment. The results, which compare the actual and the simulated daily ice evolution, are shown in Figure 10 to 13. The comparison exhibits striking similarity across the daily images, confirming that our IM preserves the more granular ice/water dynamics. 
<br/>
<br/>
<figure>
    <img src="/images/Figure10.png" width="600" height="480">
    <figcaption> Figure 10: The actual daily sea ice evolution in the focus area during a melting cycle from (a) Aug 1st to (q) Aug 16th, 2023. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure11.png" width="600" height="480">
    <figcaption> Figure 11: The simulated daily sea ice evolution, based on the semi-monthly Ising parameters, in the focus area during a melting cycle from (a) Aug 1st to (q) Aug 16th, 2023. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure12.png" width="600" height="480">
    <figcaption> Figure 12: The actual daily sea ice evolution in the focus area during a freezing cycle from (a) Oct 16th to (q) Nov 1st, 2023. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure13.png" width="600" height="480">
    <figcaption> Figure 13: The simulated daily sea ice evolution, based on the semi-month Ising parameters, in the focus area during a freezing cycle from (a) Oct 16th to (q) Nov 1st, 2023. </figcaption>
</figure>
<br/>
<br/>

<h2> 5.4	Comparison of sea ice extent between 2023 and 2012 </h2>
<br/>
2012 recorded the lowest September Arctic sea ice extent in history, while 2023 witnessed the hottest July and proved to be the hottest year. It would be an interesting experiment to compare the 2023 sea ice extent to that in 2012.
<br/>
Following the same steps as in Section 5.1, the IM simulations and ResNet50 training are conducted for the period of June 16th, 2012 to Jan 1st, 2013 for the focus area.  To keep this paper concise, we will skip the semi-monthly actual and simulated images and the ResNet50 predicted parameters. The more informative ice coverage percentage and ice extent comparison charts are nevertheless included in Figure 14. The details of 2012 results can be found in Appendix A.2. 
<br/>
<br/>
<figure>
    <img src="/images/Figure14.png" width="600" height=300">
    <figcaption>Figure 14: (a) The ice coverage percentage in the focus area from June 16th, 2012 to Jan 1st, 2013; (b) The ice extent for the same periods.  </figcaption>
</figure>
<br/>
<br/>
Comparing Figure 9 with Figure 14 indicates that 2023 did not break the record-low Arctic sea ice extent level set in 2012, validated by both the actual measures and the IM simulations. However, 2023 sets the second lowest ice extent for our focus area, below those low levels previously achieved in 2019 and 2020  (2019 and 2020 results are not included in this paper but can be provided upon request.)  Even though 2023 does not break the historical record [89], it offers no reason for us to be optimistic about the future. In fact, in the 45-year-satellite record from 1979 to 2023, 17 of the lowest minimums have all occurred in the last 17 years [90]. Many scientists are concerned that the effect of Arctic sea ice decline on global warming will intensify as the sea ice loss continues. Although predicting the sea ice extent for the future years is beyond the scope of our current study, we will discuss the possibilities and issues in the next section.
<br/><br/>


# 6. Discussions and future work
<br/>
In this paper, we introduce continuous spin values and an inertia factor to a classical 2-D IM, which is utilized to simulate the dynamics of the sea ice evolution in the Arctic region by employing the Metropolis-Hastings algorithm. Deep neural network models are trained to solve the inverse Ising problem and obtain the best-fit Ising parameters. My results show excellent similarity with the actual sea ice dynamics, based on the ice configuration images and the numerical measures including the ice coverage percentage and the ice extent. It is exciting and inspiring to see that combining the 100-year-old classical Ising model with the modern innovations in deep machine learning has the potential to bring enormous power towards climate change research and other interdisciplinary studies.
<br/>
<h2> 6.1	Discussions on the methodology </h2>
<br/>
The extrapolation ability of our generalized model is worth discussing. In other words, how does the model perform if the Ising parameters fitted from one year are applied to the data of another year? For this purpose, we conducted projection of sea ice evolution from September to December 2023 based on the 2022 best-fit parameters for the same time periods with the initial ice image on August 16th, 2023. My projection displays larger discrepancies from the actual images, since the idiosyncratic intra-year configurations are hard to reproduce by the Ising parameters from a different year. However, the ice extent metrics calculated from our experiment accurately predicts that September 2023 would record the second lowest ice extent in history for our focus area, although the extrapolation ability of the Ising parameters is far from being perfect.   
<br/>
The impact of the inertia factor I on the performance of our model is also worth discussing. In fact, we have explored the vanilla Ising model without the inertia term; the subsequent simulation results substantially underperform the results with the inertia term incorporated. This finding validates the significant strength of the inertia factor in sea ice modeling, indicating that Arctic sea ice and water indeed display the tendency to stay unchanged. However, my finding does not confirm that the inertia factor is a must-have; it is possible to improve the Ising model performance via other routes, e.g. by further enriching the functional forms of the external force B, which is out of scope of this paper.
<br/>
Details of the methodology analysis can be found in Appendix A.3 and A.4.
<br/>
<h2> 6.2	Comparison between the neural networks </h2>
<br/>
The three deep neural networks in this study, a simple CNN, a deeper fine-tuned ResNet50, and a fine-tuned ViT, all demonstrated excellent power to solve for the Ising parameters that explain the complex sea ice dynamics. However, ResNet50 marginally outperforms the other two models by delivering slightly better similarity with observations due to its greater depth to capture more complex image features. As shown in Figure 15 the results of simulated sea ice configurations for Aug 16th, 2023 and the corresponding heatmaps of absolute difference using three neural networks, simulation from all three networks exhibit good match, but the heatmap  of ResNet50 showcases smallest difference from to the actual sea ice configuration. While the ViT model can capture global relationships across image patches through its attention mechanism, the localized nature of this Ising model, where each spin influences only its immediate neighbors, makes the ViT results not as good. The ViT model, which lacks the inductive biases inherent to CNN, such as translation equivariance and locality, also requires significantly more data and computational power to train effectively [91]. The hardware (using an RTX3060 GPU) and the limited training time (for only 70 hours,) may have restricted its performance. With more training data and powerful GPUs, the ViT model has the potential to achieve better results. Models that combine CNN with transformers, such as CvT (convolutional vision transformer) [92] can also be explored in future studies.
<br/>
<br/>
<figure>
    <img src="/images/Figure15.png" width="500" height="250">
    <figcaption> Figure 15: Simulation results using three neural networks: (a) The actual sea ice configuration on Aug 16th, 2023; (b) simulation using the simple CNN; (c) simulation using ResNet50; (d) simulation using ViT; (e) heatmap illustrating the difference between the actual and the simple CNN result; (f) heatmap for ResNet50; (g) heatmap for ViT. </figcaption>
</figure>
<br/>
<br/>
<h2> 6.3	Will a “Blue Ocean Event” happen? If so, when will it be? </h2>
<br/>
Arctic sea ice extent in the most recent years was near the historic minimum recorded in 2012. As the Arctic sea ice continues to shrink, will a “Blue Ocean Event” happen, i.e., will we see an “ice-free” Arctic Ocean? Some research predicts that this can happen in the 2030s. 
<br/>
My current study will need to be extended to gain the full predictive power when utilized to answer this “Blue Ocean Event” question. As shown in Table 1, the IM parameters demonstrate the substantial impact of the external force factor B, which remains unexplored within the scope of our model. If the functional form of this external force is further enriched and linked to actual environmental factors in climate change modeling, the IM framework may prove its strength in offering the “Ising Prediction” to answer the “Blue Ocean Event” question. 
<br/>
<h2> 6.4	Quantum Ising Model </h2>
<br/>
My study sets the stage for future Ising model research on sea ice evolution. Methodologically, we generalize the classical Ising model with continuous spin values to incorporate varying ice/water percentages across the Ising lattice. A more complicated idea to be explored in future research is the Quantum Ising Model (QIM), or the so-called Transverse Field Ising Model [93]. With quantum computers, the continuous spin values can be naturally modeled by the rotation of qubits in the Bloch Sphere. Large quantum computers are inaccessible for personal usage currently; but once they are reachable, our research can be readily extended with the assistance of quantum computing in the future.
<br/><br/>

<br/><br/>
<br/><br/>
<br/><br/>







# Appendices
<br/>
<h2> A.1 Simulation results using the simple CNN and the fine-tuned ViT models </h2>
<br/>
The IM simulation results using other two neural networks—the simple CNN and the fine-tuned ViT—are described below.
<br/>
The Ising parameters $(J, B_0, B_x, B_y, I)$ for each simulation period in 2023 predicted by the simple CNN model are shown in Table 2. The simulated sea ice images for each 2023 period are shown in Figure 16 utilizing the Ising parameters in Table 2. The absolute differences in ice coverages for each of the simulation period in Figure 6 and Figure 16 are calculated; the results are illustrated as the heatmaps in Figure 17. The ice coverage percentage and ice extent comparison charts based on the simple CNN model are illustrated in Figure 18.
<br/>
<br/>
<figure>
    <img src="/images/Table2.png" width="1000" height="180">
    <figcaption> Table 2: The simple CNN model predicted Ising parameters for the 2023 sea ice evolution </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure16.png" width="600" height="400">
    <figcaption> Figure 16: The simple CNN model simulated semi-monthly sea ice evolution in the focus area in 2023. (a) is the actual image on June 16th, 2023; (b) - (n) are simulated images on (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2022, and (n) Jan 1st, 2024. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure17.png" width="600" height="400">
    <figcaption> Figure 17: Heatmaps illustrating the absolute difference in ice coverages between Figure 6 (the actual sea ice) and 16 (the simple CNN model simulated configuration) for each semi-monthly period, from (a) June 16th, 2023 to (n) Jan 1st, 2024.  Yellow color indicates a good match and red a large difference, as shown by the scale on the right. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure18.png" width="600" height="300">
    <figcaption> Figure 18: (a) The ice coverage percentage in the focus area from June 16th, 2023 to Jan 1st, 2024; (b) The ice extent for the same periods. Blue curves are the actual measures from the NRTSI data; orange ones show the IM simulation results from the simple CNN model. </figcaption>
</figure>
<br/>
<br/>
The Ising parameters $(J, B_0, B_x, B_y, I)$ for each simulation period in 2023 predicted by the fine-tuned ViT model are shown in Table 3. The simulated sea ice images for each 2023 period are shown in Figure 19 utilizing the Ising parameters in Table 3. The absolute differences in ice coverages for each of the simulation period in Figure 6 and Figure 19 are illustrated as the heatmaps in Figure 20. The ice coverage percentage and ice extent comparison charts based on the fine-tuned ViT model are illustrated in Figure 21.
<br/>
<figure>
    <img src="/images/Table3.png" width="1000" height="180">
    <figcaption> Table 3: The fine-tuned ViT model predicted Ising parameters for the 2023 sea ice evolution </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure19.png" width="600" height="400">
    <figcaption> Figure 19: The fine-tuned ViT model simulated semi-monthly sea ice evolution in the focus area in 2023. (a) is the actual image on June 16th, 2023; (b) - (n) are simulated images on (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2022, and (n) Jan 1st, 2024. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure20.png" width="600" height="400">
    <figcaption> Figure 20: Heatmaps illustrating the absolute difference in ice coverages between Figure 6 (the actual sea ice) and 16 (the fine-tuned ViT model simulated configuration) for each semi-monthly period, from (a) June 16th, 2023 to (n) Jan 1st, 2024.  Yellow color indicates a good match and red a large difference, as shown by the scale on the right. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure21.png" width="600" height="300">
    <figcaption> Figure 21: (a) The ice coverage percentage in the focus area from June 16th, 2023 to Jan 1st, 2024; (b) The ice extent for the same periods. Blue curves are the actual measures from the NRTSI data; orange ones show the IM simulation results from the fine-tuned ViT model. </figcaption>
</figure>
<br/>
<br/>
From the above results, we can see that the three deep neural networks in this study, a simple CNN, a deeper fine-tuned ResNet50, and a fine-tuned ViT, when coupled with Ising model, all demonstrated striking power explain the complex sea ice dynamics. ResNet50 marginally outperforms the other two models by delivering slightly better similarity with observations.
<br/>
<br/>
<h2> A.2 Simulation results of other years </h2>
<br/>
The IM simulation results for 2012 and 2022 based on the fine-tuned ResNet50 model are included in this section. The 2024 results will be included when the full year data is available.
<br/>
Figure 22 shows the semi-monthly NRTSI sea ice images in the focus area from June 16th, 2022 to Jan 1st, 2023. The Ising parameters $(J, B_0, B_x, B_y, I)$ for each simulation period in 2022 predicted by the ResNet50 model are shown in Table 4. The simulated sea ice images for each 2022 period are shown in Figure 23 utilizing the Ising parameters in Table 4. The absolute differences in ice coverages for each of the simulation period in Figure 22 and Figure 23 are calculated; the results are illustrated as the heatmaps in Figure 24. The ice coverage percentage and ice extent comparison charts based on the simple CNN model are illustrated in Figure 25.
<br/>
<br/>
<figure>
    <img src="/images/Figure22.png" width="600" height="400">
    <figcaption> Figure 22: The actual semi-monthly sea ice evolution in the focus area in 2022: (a) June 16th, (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2022, and (n) Jan 1st, 2023. </figcaption>
</figure>
<br/>
<br/>
<figure>
    <img src="/images/Table4.png" width="1000" height="180">
    <figcaption> Table 4: The fine-tuned ResNet50 model predicted Ising parameters for the 2022 sea ice evolution </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure23.png" width="600" height="400">
    <figcaption> Figure 23: The fine-tuned ResNet50 model simulated semi-monthly sea ice evolution in the focus area in 2022. (a) is the actual image on June 16th, 2022; (b) - (n) are simulated images on (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2022, and (n) Jan 1st, 2023. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure24.png" width="600" height="400">
    <figcaption> Figure 24: Heatmaps illustrating the absolute difference in ice coverages between Figure 22 (the actual sea ice) and 23 (the fine-tuned ResNet50 model simulated configuration) for each semi-monthly period, from (a) June 16th, 2022 to (n) Jan 1st, 2023.  Yellow color indicates a good match and red a large difference, as shown by the scale on the right. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure25.png" width="600" height="300">
    <figcaption> Figure 25: (a) The ice coverage percentage in the focus area from June 16th, 2022 to Jan 1st, 2023; (b) The ice extent for the same periods. Blue curves are the actual measures from the NRTSI data; orange ones show the IM simulation results from the fine-tuned ResNet50 model. </figcaption>
</figure>
<br/>
<br/>
Figure 26 shows the semi-monthly NRTSI sea ice images in the focus area from June 16th, 2012 to Jan 1st, 2013. The Ising parameters $(J, B_0, B_x, B_y, I)$ for each simulation period in 2012 predicted by the ResNet50 model are shown in Table 5. The simulated sea ice images for each 2012 period are shown in Figure 27 utilizing the Ising parameters in Table 5. The absolute differences in ice coverages for each of the simulation period in Figure 26 and Figure 27 are calculated; the results are illustrated as the heatmaps in Figure 28. The ice coverage percentage and ice extent comparison charts based on the simple CNN model are illustrated in Figure 29.
<br/>
<br/>
<figure>
    <img src="/images/Figure26.png" width="600" height="400">
    <figcaption> Figure 26: The actual semi-monthly sea ice evolution in the focus area in 2012: (a) June 16th, (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2012, and (n) Jan 1st, 2013. </figcaption>
</figure>
<br/>
<br/>
<figure>
    <img src="/images/Table5.png" width="1000" height="180">
    <figcaption> Table 5: The fine-tuned ResNet50 model predicted Ising parameters for the 2012 sea ice evolution </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure27.png" width="600" height="400">
    <figcaption> Figure 27: The fine-tuned ResNet50 model simulated semi-monthly sea ice evolution in the focus area in 2012. (a) is the actual image on June 16th, 2012; (b) - (n) are simulated images on (b) July 1st, (c) July 16th, (d) Aug 1st, (e) Aug 16th, (f) Sept 1st, (g) Sept 16th, (h) Oct 1st, (i) Oct 16th, (j) Nov 1st, (k) Nov 16th, (l) Dec 1st, (m) Dec 16th, 2012, and (n) Jan 1st, 2013. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure28.png" width="600" height="400">
    <figcaption> Figure 28: Heatmaps illustrating the absolute difference in ice coverages between Figure 26 (the actual sea ice) and 27 (the fine-tuned ResNet50 model simulated configuration) for each semi-monthly period, from (a) June 16th, 2012 to (n) Jan 1st, 2013.  Yellow color indicates a good match and red a large difference, as shown by the scale on the right. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure29.png" width="600" height="300">
    <figcaption> Figure 29: (a) The ice coverage percentage in the focus area from June 16th, 2012 to Jan 1st, 2013; (b) The ice extent for the same periods. Blue curves are the actual measures from the NRTSI data; orange ones show the IM simulation results from the fine-tuned ResNet50 model. </figcaption>
</figure>
<br/>
<br/>
<h2> A.3 Methodology analysis: the effect of the inertia factor  </h2>
<br/>
The impact of the inertia factor I on the performance of our model is worth investigation. We have explored the vanilla Ising model without the inertia term; the subsequent simulation results for 2022 are illustrated in Table 6 and Figure 30 to Figure 32. It can be seen that they substantially underperform the results with the inertia term incorporated. Adding the inertia factor makes the simulation process much more robust, validating that this added feature has significant strength in sea ice modeling. Intuitively, this finding indicates that Arctic sea ice and water have a tendency to stay unchanged even in the presence of external forces.
<br/>
<br/>
<figure>
    <img src="/images/Table6.png" width="900" height="140">
    <figcaption> Table 6: The predicted Ising parameters for the 2012 sea ice evolution without the inertia factor I. </figcaption>
</figure>
<br/>
<br/>
<figure>
    <img src="/images/Figure30.png" width="600" height="400">
    <figcaption> Figure 30: The simulated semi-monthly evolution of sea ice for our focus area in 2022, using the Ising model without the inertia factor. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure31.png" width="600" height="400">
    <figcaption> Figure 31: Heatmaps illustrating the difference in the ice coverage between Figure 22 (the 2022 actual sea ice) and figure 30 (the simulation based on Ising model without the inertia factor). </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure32.png" width="600" height="300">
    <figcaption> Figure 32: The ice coverage percentage and the ice extent of 2022, where the simulated curves are based on the Ising model without the inertia term. </figcaption>
</figure>
<br/>
<br/>
<br/>
<h2> A.4 Methodology analysis: extrapolation power of the model </h2>
<br/>
This section addresses the extrapolation capability of this generalized Ising model. How does the model perform if the Ising parameters fitted from one year are applied to the data of a different year? For this purpose, projection of sea ice evolution from September to December 2023 has been conducted based on the 2022 best-fit parameters (as in Table 7 which is copied from part of Table 4) for the same time periods with the initial ice image on August 16th, 2023. Figure 33 Shows the actual sea ice evolution from Aug 16th, 2023 to Jan 1st, 2024; Figure 34 shows the IM simulated evolution based on 2022 parameters from Table 4; Figure 35 Shows the ice coverage percentage and extent based on Figure 33 and Figure 34. Notably, although larger deviations are observed from the actual images, because the idiosyncratic intra-year configurations are hard to reproduce by the Ising parameters from a different year. However, the ice extent metrics calculated from my experiment accurately predict that September 2023 would record the second lowest ice extent in history for my focus area. The extrapolation ability of our model, even though far from being perfect, is strong and can offer many insights into the sea ice dynamics for the near future.
<br/>
<br/>
<figure>
    <img src="/images/Table7.png" width="700" height="180">
    <figcaption> Table 7:  The 2022 partial year Ising parameters copied from Table 4, used for extrapolation analysis. </figcaption>
</figure>
<br/>
<br/>
<figure>
    <img src="/images/Figure33.png" width="600" height="300">
    <figcaption> Figure 33: The actual semi-monthly evolution of sea ice from (a) Aug 16th, 2023 to (j) Jan 1st, 2024.  </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure34.png" width="600" height="300">
    <figcaption> Figure 34: The simulated semi-monthly evolution of sea ice in our focus area in the near future. (a) is the actual image on Aug 16th, 2023 as the start state; (b)-(j) are simulated images (based on the best-fit IM parameters in the 2022 simulations over the corresponding semi-monthly periods) from (b) Sept 1st, 2023 to (j) Jan 1st, 2024. </figcaption>
</figure>
<br/>
<br/>
<br/>
<figure>
    <img src="/images/Figure35.png" width="600" height="300">
    <figcaption> Figure 35: (a) The ice coverage percentage and (b) the sea ice extent for our focus area in 2023. The predicted (orange) curves before Sept 1st, 2023 are based on 2023 best-fit parameters; from Sept 1st, 2023 onwards are based on 2022 best-fit parameters. </figcaption>
</figure>
<br/>

<br/><br/>
<br/><br/>
<br/><br/>

# Code list
The code files for this study include: <br/>
• download_NSIDC.py<br/>
• ReadSeaIce.py<br/>
• IIMConstants.py<br/>
• IceIsingCont.py<br/>
• IIMCNNModel.py<br/>
• IIMCNNRun.py<br/>
• IIM_CNN_Results.py<br/>
• IIMResnetFineTune.py<br/>
• IIM_Resnet_Results.py<br/>
• IIMViTFineTune.py<br/>
• IIM_ViT_Results.py<br/>
• IIMSimul.py<br/>
• Gen_Figures.py<br/>

<br/>
Detailed desription of the code is in CodeList.txt
<br/><br/>


# Bibliography <br/>
[1] 	National Snow and Ice Data Center, "Sea Ice Today," 2024. [Online]. Available: https://nsidc.org/sea-ice-today. <br/>
[2] 	NOAA National Centers for Environmental Information, "2023 was the warmest year in the modern temperature record," Januray 2024. [Online]. Available: https://www.climate.gov/news-features/featured-images/2023-was-warmest-year-modern-temperature-record#:~:text=The%20year%202023%20was%20the,decade%20(2014%E2%80%932023).<br/>
[3] 	E. Wang, "A study on Arctic sea ice dynamics using the continuous spin Ising model," Journal of Applied Physics, vol. 135, p. 194901, 2024. <br/>
[4] 	E. Ising, "Beitrag zur Theorie des Ferromagnetismus," Z. Phys, vol. 31, no. 1, p. 2530258, 1925. <br/>
[5] 	E. Ising, Contribution to the Theory of Ferromagnetism, 1924. <br/>
[6] 	S. G. Brush, "History of the Lenz-Ising model," Review of Modern Physics, vol. 39, no. 4, p. 883, 1967. <br/>
[7] 	L. Onsager, "Crystal statistics. I. A two-dimensional model with an order-disorder transition," Physical Review, vol. 65, no. 3-4, pp. 117-149, 1944. <br/>
[8] 	H. A. Kramers and G. H. Wannier, "Statistics of the Two-Dimensional Ferromagnet. Part I," Physical Review, vol. 60, no. 3, pp. 252-262, 1941. <br/>
[9] 	H. A. Kramers and G. H. Wannier, "Statistics of the Two-Dimensional Ferromagnet. Part II," Physical Review, vol. 60, no. 3, pp. 263-176, 1941. <br/>
[10] 	J. Zuber and C. Itzykson, "Quantum field theory and the two-dimensional Ising model," Physical Review D, vol. 15, p. 2875, 1977. <br/>
[11] 	M. Aguilera, S. A. Moosavi and H. Shimazaki, "A unifying framework for mean-field theories," Nature Communications, vol. 12, p. 1197, 2021. <br/> 
[12] 	S. Sides, P. Rikvold and M. Novotony, "Kinetic Ising model in an oscilating field: finite-size scaling at the dynamic phase transition," Physical review letters, vol. 81, no. 4, p. 4865, 1998. <br/>
[13] 	D. Stauffer, "Social applications of two-dimensional Ising models," American Journal of Physics, vol. 76, no. 4, pp. 470-473, 2008. <br/>
[14] 	C. Campajola, F. Lillo and D. Tantari, "Inference of the kinetic Ising model with heterogeneous missing data," Physical Review E, vol. 99, no. 6, p. 062138, 2019. <br/>
[15] 	B. Dun and Y. Roudi, "Learning and inference in a nonequilibrium Ising model with hidden nodes," Physical Review E, vol. 87, no. 2, p. 022127, 2013. <br/>
[16] 	N. N. Vtyurina, D. Dulin, M. W. Docter, A. S. Meyer, N. H. Dekker and E. A. Abbondanzieri, "Hysteresis in DNA compaction by Dps is described by an Ising model," Proceedings of the National Academy of Sciences, vol. 113, no. 18, pp. 4982-4987, 2016. <br/>
[17] 	J. Majewski, H. Li and J. Ott, "The Ising model in physics and statistical genetics," The American Journal of Human Genetics, vol. 69, no. 4, pp. 853-862, 2001. <br/>
[18] 	A. Witoelar and Y. Roudi, "Neural network reconstruction using kinetic Ising models with memory," BMC Neurosci., vol. 12, p. 274, 2011. <br/>
[19] 	C. Donner and M. Opper, "Inverse Ising problem in continuous time: a latent variable approach," Physical Review E, vol. 96, p. 061104, 2017. <br/>
[20] 	J. Hertz, Y. Roudi and J. Tyrcha, "Ising model for inferring network structure from spike data," arXiv.1106.1752, 2011. <br/>
[21] 	Y. Roudi, D. B. and J. Hertz, "Multi-neuronal activity and functional connectivity in cell assemblies," Curr. Opin. Neurobiol, vol. 32, p. 38, 2015. <br/>
[22] 	Y. Shi and T. Duke, "Cooperative model of bacteril sensing," Physical Review E, vol. 58, no. 5, pp. 6399-6406, 1998. <br/>
[23] 	T. F. Stepinski, "Spatially explicit simulation of deforestation using the Ising-like neutral model," Environmental Research: Ecology, vol. 2, no. 2, p. 025003, 2023. <br/>
[24] 	T. F. Stepinski and J. Nowosad, "The kinetic Ising model encapsulates essential dynamics of land pattern change," Royal Society Open Science, vol. 10, no. 10, p. 231005, 2023. <br/>
[25] 	Y.-P. Ma, I. Sudakov, C. Strong and K. Golden, "Ising model for melt ponds on Arctic sea ice," New Journal of Physics, vol. 21, p. 063029, 2019. <br/>
[26] 	T. C. Schelling, "Dynamic models of segregation," J. Math. Sociol., vol. 1, pp. 143-186, 1971. <br/>
[27] 	J. P. Bouchaud, "Crises and collective socio-economic phenomena: simple models and challenges," J. Stat. Phys., vol. 151, pp. 567-606, 2013. <br/>
[28] 	S. Bornholdt and F. Wagner, "Stability of money: phase transitions," Physica A: Statistical Mechanics and its Applications, vol. 316, no. 1-4, pp. 453-468, 2002. <br/>
[29] 	G. E. Hinton, "How Neural Networks Learn from Experience," Scientific American, vol. 267, no. 3, pp. 144-151, 1992. <br/>
[30] 	D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning representations by back-propagating errors," Nature, vol. 323, pp. 533-536, 1986. <br/>
[31] 	J. Schmidhuber, "Annotated History of Modern AI and Deep Learning," arXiv, vol. 2212, p. 11279, 2022. <br/>
[32] 	J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities," The Proceedings of the National Academy of Sciences, vol. 79, no. 8, pp. 2554-2558, 1982. <br/>
[33] 	Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard and L. D. Jackel, "Backpropagation Applied to Handwritten Zip Code Recognition," Neural Computation, vol. 1, no. 4, pp. 541-551, 1989. <br/>
[34] 	K. Fukushima, "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position," Biological Cybernetics, vol. 36, pp. 193-202, 1980. <br/>
[35] 	A. Waibel, "Phoneme Recognition Using Time-Delay Neural Networks," in Meeting of IEICE, Tokyo, Japan, 1987. <br/>
[36] 	D. Ciresan, U. Meier, J. Masci, L. M. Gambardella and J. Schmidhuber, "Flexible, High Performance Convolutional Neural Networks for Image Classification," in Proceedings of the Twenty-Second International Joint Conference on Artificial Intelligence, 2011. <br/>
[37] 	A. Tsantekidis, N. Passalis, A. Tefas, J. Kanniainen, M. Gabbouj and A. Iosifidis, "Forecasting Stock Prices from the Limit Order Book Using Convolutional Neural Networks," in IEEE 19th Conference on Business Informatics (CBI), 2017. <br/>
[38] 	A. van den Oord, S. Dieleman and B. Schrauwen, "Deep content-based music recommendation," in Advances in Neural Information Processing Systems 26 (NIPS 2013), 2013. <br/>
[39] 	R. Collobert and J. Weston, "A unified architecture for natural language processing," in Proceedings of the 25th international conference on Machine learning - ICML, New York, USA, 2008. <br/>
[40] 	A. Krizhevsky, I. Sutskever and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," in Advances in Neural Information Processing Systems 25 (NIPS 2012), 2012. <br/>
[41] 	K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," arXiv:1409.1556, 2014. <br/>
[42] 	C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke and A. Rabinovich, "Going deeper with convolutions," in Conference on Computer Vision and Pattern Recognition (CVPR), 2015. <br/>
[43] 	K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," in Conference on Computer Vision and Pattern Recognition (CVPR), 2016. <br/>
[44] 	G. Huang, Z. Liu, L. van der Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," arXiv:1608.06993, 2016. <br/>
[45] 	M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," arXiv:1905.11946, 2019. <br/>
[46] 	A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser and I. Polosukhin, "Attention is All you Need," in Advances in Neural Information Processing Systems, 30, 2017. <br/>
[47] 	S. Hochreiter and J. Schmidhuber, "Long Short-term Memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997. <br/>
[48] 	K. Cho, B. van Merrienboer, D. Bahdanau, F. Bougares, H. Schwenk and Y. Bengio, "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," arXiv:1406.1078, 2014. <br/>
[49] 	OpenAI, J. Achiam, S. Adler, S. Agarwal and e. al, "GPT-4 Technical Report," arXiv:2303.08774, 2023. <br/>
[50] 	T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu and C. Xu, "Transformers: State-of-the-Art Natural Language Processing," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020. <br/>
[51] 	J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805, 2018. <br/>
[52] 	H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave and G. Lample, "LLaMA: Open and Efficient Foundation Language Models," arXiv:2302.13971, 2023. <br/>
[53] 	A. Ruoss, G. Delétang, S. Medapati, J. Grau-Moya, L. Wenliang, E. Catt, J. Reid and T. Genewein, "Grandmaster-Level Chess Without Search," arXiv:2402.04494v1, 2024. <br/>
[54] 	M. Ciolino, D. Noever and J. Kalin, "The Go Transformer: Natural Language Modeling for Game Play," arXiv:2007.03500, 2020. <br/>
[55] 	P. Xu, X. Zhu and D. A. Clifton, "Multimodal Learning with Transformers: A Survey," arXiv:2206.06488v2, 2023. <br/>
[56] 	P. Ramachandran, N. Parmar, A. Vaswani, I. Bello, A. Levskaya and J. Shlens, "Stand-Alone Self-Attention in Vision Models," in Advances in Neural Information Processing Systems 32, 2019. <br/>
[57] 	A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly and J. Uszkoreit, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," arXiv:2010.11929, 2021. <br/>
[58] 	United States Environmental Protection Agency, "Climate Change Indicators: Arctic Sea Ice," 2021. [Online]. Available: https://www.epa.gov/climate-indicators/climate-change-indicators-arctic-sea-ice#:~:text=September%20is%20typically%20when%20the,maximum%20extent%20after%20winter%20freezing. <br/>
[59] 	NASA Langley Research Center's Atmospheric Science Data Center, "Ice-Albedo Feedback in the Arctic," NASA, 2020. <br/>
[60] 	K. L. Oakley, M. E. Whalen, D. C. Douglas, M. S. Udevitz, T. C. Atwood and C. Jay, "Polar bear and walrus response to the rapid decline in Arctic sea ice," USGS - Science for a changing world, 2012. [Online]. Available: https://pubs.usgs.gov/publication/fs20123131.<br/>
[61] 	NSIDC, "EASE-Grid sea ice age, version 4," 2021. <br/>
[62] 	Copernius, "Record warm November consolidates 2023 as the warmest year," 2023. [Online]. Available: https://climate.copernicus.eu/record-warm-november-consolidates-2023-warmest-year#:~:text=The%20extraordinary%20global%20November%20temperatures,Climate%20Change%20Service%20(C3S). <br/>
[63] 	NYTimes, "2024 on Track to Be the Hottest Year on Record," August 2024. [Online]. Available: https://www.nytimes.com/2024/08/08/climate/heat-records-2024.html. <br/>
[64] 	A. Shekaari and M. Jafari, "Theory and simulation of the Ising model," arXiv, 2021. <br/>
[65] 	G. S. Sylvester and H. van Beijeren, "Phase Transitions for Continous-Spin Ising Ferromagnets," Journal of Functional Analysis, vol. 28, pp. 145-167, 1978. <br/>
[66] 	E. Bayong and H. T. Diep, "Effect of long-range interactions on the critical behavior of the continuous Ising model," Physical Review B, vol. 59, no. 18, p. 11919, 1999. <br/>
[67] 	M. Krasnytska, B. Berche, Y. Holovatch and R. Kenna, "Ising model with variable spin/agent strengths," Journal of Physics: Complexity, vol. 1, p. 035008, 2020. <br/>
[68] 	P. Basua, J. Bhattacharya, D. P. S. Jakkab, C. Mosomane and V. Shukla, "Machine learning of Ising criticality with spin-shuffling," arXiv, 2023. <br/>
[69] 	N. Walker, K. Tam and M. Jarrell, "Deep learning on the 2 dimensional Ising model to extract the crossover region with a variational autoencoder," Scientific Reports, vol. 10, p. 13047, 2020. <br/>
[70] 	G. S. Hartnett and M. Mohseni, "Self-Supervised Learning of Generative Spin-Glasses with Normalizing Flows," arXiv, 2020. <br/>
[71] 	J. Albert and R. H. Swendsen, "The Inverse Ising Problem," Physics Procedia, vol. 57, pp. 99-103, 2014. <br/>
[72] 	"Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)," image-net.org, 2012. [Online]. Available: https://image-net.org/challenges/LSVRC/2012/results.html. <br/>
[73] 	H. Mhaskar, Q. Liao and T. Poggio, "When and Why Are Deep Networks Better than Shallow Ones?," in Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, 2017. <br/>
[74] 	Y. Bengio, P. Simard and P. Frascon, "Learning long-term dependencies with gradient descent is difficult," IEEE Transactions on Neural, vol. 5, no. 2, pp. 157-166, 1994. <br/>
[75] 	X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, 2010. <br/>
[76] 	K. He and J. Sun, "Convolutional Neural Networks at Constrained Time Cost," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. <br/>
[77] 	F. Rosenblatt, "The Perceptron: A Probabilistic Model For Information Storage And Organization in the Brain," Psychological Review, vol. 65, no. 6, pp. 386-408, 1958. <br/>
[78] 	W. N. Meier, J. S. Stewart, H. Wilcox, M. A. Hardman and D. Scott, "Near-Real-Time DMSP SSMIS Daily Polar Gridded Sea Ice Concentrations, Version 2," NASA National Snow and Ice Data Center Distributed Active Archive Center, Boulder, Colorado USA, 2023. <br/>
[79] 	T. F. Stepinski and J. Nowosad, "The kinetic Ising model encapsulates essential dynamics of land pattern change," Royal Society Open Science, vol. 10, p. 231005, 2023. <br/>
[80] 	Google Brain Team, "Keras: The high-level API for TensorFlow," [Online]. Available: https://www.tensorflow.org/guide/keras. <br/>
[81] 	J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and F.-F. Li, "ImageNet: A large-scale hierarchical image database," in Conference on Computer Vision and Pattern Recognition (CVPR), Miami, FL, USA, 2009. <br/>
[82] 	A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga and A. Lerer, "Automatic differentiation in PyTorch," in 31st Conference on Neural Information Processing Systems, 2017. <br/>
[83] 	"ResNet," [Online]. Available: https://pytorch.org/vision/stable/models/resnet.html. <br/>
[84] 	H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles and H. Jégou, "Training data-efficient image transformers & distillation through attention," arXiv:2012.12877, 2021. <br/>
[85] 	H. Bao, L. Dong, S. Piao and F. Wei, "BEiT: BERT Pre-Training of Image Transformers," arXiv:2106.08254, 2022. <br/>
[86] 	"google/vit-base-patch16-224," https://huggingface.co/google/vit-base-patch16-224, [Online]. Available: https://huggingface.co/google/vit-base-patch16-224. <br/>
[87] 	"Transformers," Hugging Face, [Online]. Available: https://huggingface.co/docs/transformers/en/index. <br/>
[88] 	"Hugging Face," [Online]. Available: https://huggingface.co/. <br/>
[89] 	O. M. Johannessen and T. I. Olaussen, "Arctic sea-ice extent: No record minimum in 2023 or recent years," Atmospheric and Oceanic Science Letters, p. 100499, 2024. <br/>
[90] 	"Arctic sea ice minimum at sixth lowest extent on record," National Snow & Ice Data Center,, 2023. [Online]. Available: https://nsidc.org/arcticseaicenews/2023/09/arctic-sea-ice-minimum-at-sixth/. <br/>
[91] 	Z. Lu, H. Xie, C. Liu and Y. Zhang, "Bridging the gap between vision transformers and convolutional neural networks on small datasets," in NIPS: Proceedings of the 36th International Conference on Neural Information Processing Systems, 2022. <br/>
[92] 	H. Wu, B. Xiao, N. Codella, M. Liu, X. Dai, L. Yuan and L. Zhang, "CvT: Introducing Convolutions to Vision Transformers," arXiv:2103.15808, 2021. <br/>
[93] 	B. K. Chakrabarti, A. Dutta and P. Sen, Quantum Ising Phases and Transitions in Transverse Ising Models, Berlin: Springer, 1996. <br/>




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


