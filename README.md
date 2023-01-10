<h1>Overview</h1>
This repository is the implementation of the paper <strong>"Activity Map and Transition Pathways of G Protein Coupled Receptor Revealed by Machine Learning"</strong>. This project uses shallow ML models to predict activity level and activation state of a given protein. Furthermore, we applied the model on MD trajectories of &beta;<sub>2</sub>AR receptor to correlate conformational structures of the protein to the activity levels. The simulations contain different conformational states of the protein including agonist, apo, and inverse agonist.

Find the database for agonist conformational state of &beta;<sub>2</sub>AR receptor here: [here](https://cmu.box.com/s/w957ph9dbdstcrrqfp7tjyv34wt4otq7)\
Find the database for apo conformational state of &beta;<sub>2</sub>AR receptor here: [here](https://cmu.box.com/s/b7son6ubfljbsfxl8wn7h0mi68pjlmvw)\
Find the database for inverse agonist conformational state of &beta;<sub>2</sub>AR receptor here: [here](https://cmu.box.com/s/o9jmpit3w45c5hseafr8ehiifaqrbheb)

<h3>Training ML models</h3>
The ML models are trained with contanct distances of amino acids engaged in the Polar Network and anlges of NPY aminos acids included in the NPxxY motif. To measure the features of GPCRs in the training dataset the <code>TRAINSET_pn_cont_npy_angl.py'</code> script in this repository should be used.

To run the script all necessary information are provided in this repository as follows:
<ul>
<li> <strong>TM_only.zip</strong> contains 555 GPCRs spatially aligned to NTSR1 receptors used for training shallow ML models. It should be unzipped before running the script.</li>
<li> <strong>Labels_GPCRdb.csv</strong> provides activation states and percentage of activity level of each of the receptors in the training dataset.</li>
<li> <strong>template.pdb</strong> is the protein data bank of NTSR1 receptor used as the reference receptor.</li>
<li> For training ML models, a few of receptors in the dataset may not contain all the required amino acids for the defined features. To compensate the missed measurements they are replaced based on label of activation of the receptor using <em>inact_int_act_55_cont_PN_for_incomplete_receptors.pkl</em> file for the contact distances and <em>inact_int_act_NPY_anles_for_incomplete_receptors.pkl</em> file for the angle features. </li>
</ul>
The results for the features of the GPCRs in the training dataset can be found here <code>Training_features_in_555_proteins.npy</code>. 

<h3>Activation prediction accuracy of ML models</h3>
<code>ML_models_activation_prediction_accuracies.py</code> implements the performance of shallow ML models including Random Forest, Decision Tree, and XGBoost in both classification and regression tasks. The <code>Training_features_in_555_proteins.npy</code> can be used to get the accuracies of the ML models trained with the defined features of the GPCRs in the dataset

<h3>Testset measurements</h3>
In order to measure all the features for testset the <code>TESTSET_pn_cont_npy_angl.py</code> script should be used. In the script <code>res_indx</code> introduces the indices of amino acids engaged in the Polar Network. If protein in the simulations is &beta;<sub>2</sub>AR do not change the indices. 






