<h1>Overview</h1>
This repository is the implementation of the paper "Activity Map and Transition Pathways of G Protein Coupled Receptor Revealed by Machine Learning". This project uses shallow ML models to predict activity level and activation state of a given protein. To present the performance of the ML models we used MD trajectories of Beta2AR receptors in different conformational states including agonist, apo, and inverse agonist.

Find the database for agonist conformational state of beta2AR receptor here: [here](https://cmu.box.com/s/w957ph9dbdstcrrqfp7tjyv34wt4otq7)\
Find the database for apo conformational state of beta2AR receptor here: [here](https://cmu.box.com/s/b7son6ubfljbsfxl8wn7h0mi68pjlmvw)\
Find the database for inverse agonist conformational state of beta2AR receptor here: [here](https://cmu.box.com/s/o9jmpit3w45c5hseafr8ehiifaqrbheb)

<h3>Training ML models</h3>
A training dataset is required in order to train ML models with contanct distances of amino acids engages in the Polar Network and anlges of NPY aminos acids included in the NPxxY motif. To measure the features in the 555 GPCRs <code>TRAINSET_55_PN_cont_NPY_angle_convertible_for_TESTSET.py'</code> script in this repository should be used.

To run the script all necessary information are provided in this repository as follows:
<ul>
<li> <strong>TM_only.zip</strong> contains 555 GPCRs spatially aligned to NTSR1 receptors used for training shallow ML models. It should be unzipped before running the script.</li>
<li> <strong>Labels_GPCRdb.csv</strong> provides labels of activation states and percentage of activity level of each of the receptors in the dataset.</li>
<li> <strong>template.pdb</strong> is the protein data bank of NTSR1 receptor that used as the reference receptor.</li>
<li> For training ML models, 58 features are used and a few of receptors in the dataset may not contain all the amino acids. To compensate the missed measurements they are replaced based on label of activation of the receptor using <em>inact_int_act_55_cont_PN_for_incomplete_receptors.pkl</em> for the contact distances and <em>inact_int_act_NPY_anles_for_incomplete_receptors.pkl</em> for the angle features. </li>
</ul>
In order to measure all the features for testset the <code>TRAINSET_55_PN_cont_NPY_angle_convertible_for_TESTSET.py'</code> script should be edited. The required cells are 1, 7, 8, 11, 12, 13, 14 and all the <code>try:</code> and <code>except:</code> should be removed in cell 11 because the testset are not labeled. 





