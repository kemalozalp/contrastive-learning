# contrastive-learning
A contrastive learning encoder inspired by Google's "Supervised Contrastive Learning" paper.
I used the two papers from Google below:
* Unsupervised Contrastive Learning: http://proceedings.mlr.press/v119/chen20j/chen20j.pdf
* Supervised Contrastive Learning: https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf

To train the encoder, follow the steps below:
1. Clone the repo
2. Change your working directory to contrastive-learning folder.
3. Create the conda environment or manually install the packages in the env.yml. If conda is installed on your system, use "conda env create --name myevn --file env.yml. A friendly warning: Conda can throw some warning messages. Ignore them and give it some time. It should resolve conflicts and setup the env properly. The env.yml was created using the --from-history flag to make it cross-platform compatible.
4. Create two folders under data: 02_intermediate and 03_processed. These folders will be used to put data_int.csv and train.pq and val.pq
5. Run all cells in "preprocessing.ipynb". This will clean the raw dataset and create Extended Connectivity Fingerprints from the SMILES with proper cleanup of compounds.
6. Run all cells in "train_contrastive_learning_encoder.ipynb". You should see the loss declining indicating the encoder trains properly. Note that the encoder might be overtraining and perform badly on the validation set when used for classificaiton. However, the goal of this example is not to train a good contrastive learning classifier but demonstrating an example of contrastive learning model in cheminformatics.