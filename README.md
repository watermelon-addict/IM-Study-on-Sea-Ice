
<!---
!["Figure 1: Arctic sea ice and the focus area"](/images/Figure1.jpg)
-->

<figure>
    <img src="/images/Figure1.jpg">
    <figcaption>Figure 1: Arctic sea ice and the focus area</figcaption>
</figure>

<br/><br/>

!["Figure 2: CNN architecture"](https://github.com/Watermelon-Addict/IM-Study-on-Sea-Ice/assets/160803085/f472882a-8849-41d6-b456-130cac58b860)

<br/><br/>

1. Full list of the Python code files
The code files this project uses when employing an innovative Continuous Spin Ising Model and Convolutional Neural Networks to study the dynamics of Arctic Sea Ice include:  
•	download_NSIDC.py
•	ReadSeaIce.py
•	IIMConstants.py
•	IceIsingCont.py
•	IIMCNNModel.py
•	IIMCNNRun.py
•	IIM_CNN_Results.py
•	IIMSimul.py
•	Gen_Figures.py.




2. The functionalities of the files

The backbone of the model is coded in IceIsingCont.py and IIMCNNModel.py. 
•	IceIsingCont.py sets up the Ising lattice and the Monte Carlo simulation process. 
•	IIMCNNModel.py creates the convolutional neural network model and sets up the training process following the Keras/Tensorflow CNN paradigm for image recognition (ref. https://www.tensorflow.org/tutorials/images/cnn). This file also contains the functions to generate CNN training datasets.

Other files are for either one of the following purposes:
•	Data downloading, processing, and other auxiliary functions/constants, including download_NSIDC.py, ReadSeaIce.py, and IIMConstants.py
•	Calls to the functions in the 2 backbone files to run actual simulation and training processes, or functions to collect and plot the result, including  IIMCNNRun.py, IIMSimul.py, IIM_CNN_Results.py, and Gen_Figures.py





3. Detailed description of all files:

download_NSIDC.py: 	Batch download daily sea ice observations .nc file, provided by NSIDC publicly.

ReadSeaIce.py:		Read NSIDC sea ice .nc file to a 2-D array; load our focus area and normalize values. 
			This script also includes a function avg_freeze() which calculates the average ice level of the lattice.

IIMConstants.py: 	Ice Ising model constants shared across multiple scripts in this project.

IceIsingCont.py:	Ising lattice class which is the foundation of this project. This script is based on a public github project: https://github.com/red-starter/Ising-Model, then it is revised by adding continuous spin values, the inertia factor in the Metropolis MC simulation, and enhanced external file functional form.
			__init__(): 	Initialize Ising lattice and the interaction parameters
			E_elem(): 	Calculate Hamiltonian of one cell
			E_tot(): 	Calculate Hamiltonian of full lattice
			metropolis():	Metropolis Monte Carlo Simulation

IIMCNNModel.py:		The main script of functions on convolutional neural network model setup and training, based on Keras/Tensorflow machine learning framework. Functions in this script are called by IIMCNNRun.py to solve the Inverse Ising problem.
Functions include:
			createModel():  Set up the CNN model with 2 chanels in input data, 4 convolutional + maxpooling layers, followed by fully connected layers with dropout for overfitting prevention.
			LoadData():	load full year of training inputs data from ..\\IIMParamGen\\IIMParamGen_{:s}.json.
			CNNTrain(): 	Train the CNN model to find the Ising parameters, save the trained model in "IIM_CNN_modelxx". If check==True, then display the output lattice to compare with the period end date observation.
			SinglePeriodGen(): Generate training data for a single period. Save the metropolis results ..\\IIMParamGen\\IIMParamGen_{:s}.json
			MultiPeriodGen(): Generate traning data for multi periods of a full year. These are used as inputs for CNN model training.
			YMD_start(): 	Auxiliary function to return the start date of all periods of a year
			YMD_end(): 	Auxiliary function to return the start date of all periods of a year

IIMCNNRun.py: 		This script calls functions in IIMCNNModel.py to train and save the IIM_CNN model. Generate random Ising parameters and run metropolis simulation, save the simulation results in ..\\IIMParamGen\\IIMParamGen_{:s}.json. These results are used as inputs for CNN model training. The code runs in parallel; and it takes hours to run so be cautious to run.

IIM_CNN_Results.py:	Functions to collect/save the annual Ising/CNN results.
			save_CNN_Res():	Save CNN Ising parameters results and the lattice for a single period
			collect_CNN_res(): Collect multiple periods of a year, save CNN results

IIMSimul.py:		Codes for most of the auxiliary functions including generating plots for the paper, testing functions for simulation results, comparisons of ice extents across years, intermediate results using different optimization method dual_annealing, and prediction of future ice extent performed in summer of 2023, etc. The functions in this script are called by IIMCNNModel.py and Gen_Figures.py.
			Functions directly used for the paper:
    			IIM_period_test():	Tests a period with certain IM parameters and displays images.
    			read_result():		Reads Ising model simulation result
			load_year():		Loads a full year of results and plots.
    			day_by_day(): 		Displays daily simultion vs. observation
    			extent_avg_plot():	Displays the avg and extent of observations and simulations. Argument "pf_date" is only used for Aug2023 project future.
    
			Some functions in this script are used for intermediate results or debugging purpose, which include:        						project_future(): 	Test in Aug2023 to predict ice extent for the following months of 2023
    			extent_avg_pred_comp():	Compare Aug2023 prediction vs. observations of Sep/Oct/Nov 2023.
    			extent_avg_prev_years():Auxiliary functions to plot extent avg across different years
			IIM_cost_diff():	Calulation for target cost function for dual_annealing optimization.	
    			IIM_period_cost():	Target cost function of a single period optimization.
    			single_run():		Estimation of Ising parameters based for single period using dual_annealing optimization.
    			annual_run(): 		Estimation of Ising parameters based for all periods of a year in parallel.
    
Gen_Figures.py:		Collect final results and plot the figures used in the paper.

IIM_CNN.png:		The diagram of the CNN model architecture used in this project, plotted by IIMCNNModel.py.




4. Copyright for download_NSIDC.py:
# Copyright (c) 2023 Regents of the University of Colorado
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.


