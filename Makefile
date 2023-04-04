create_env:
	conda create -n mros_data python=3.10 numpy scikit-learn pytest flake8 black ruby
	conda install -n mros_data -c conda-forge mne librosa rich ipympl
	conda install -n mros_data pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	conda install -n mros_data pytorch-lightning -c conda-forge
