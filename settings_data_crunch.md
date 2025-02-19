## Settings VM DataCrunch before starting

After having the VM setted up, we need:

* Secure the instances with: 
```
sudo apt update
sudo apt install fail2ban
sudo systemctl start fail2ban
sudo systemctl enable fail2ban
sudo apt install ufw
sudo ufw allow ssh
sudo ufw enable
```

* Install Conda Miniforge
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

* Install git lfs and GitHub cli
```
sudo apt update
sudo apt install git-lfs
sudo apt install gh
```

* Log into github.com 
```
gh auth login
```

## Importing MIMICIV data from local PC
* Import the Data From DC:
instead of continuously paying the storage space (which has a cost), for preprocessing purpose we can import in the 
VM environment the data stored on my laptop using scp. Create a folder /data (place it in git ignore since we do not
want to export data on git-hub)
```
mkdir data
echo "data/" >> .gitignore
```

To transfer mimiciv data from my local pc to the VM environment, from the local terminal:
```
scp -r ~/Documents/PHD_LLM/MIMICIV/mimic-iv-3.1 <root@192.168.1.100>:/root/MIMICIV/data/
```

* Unzip the data in the WM
The data is zipped once in the VM. To unzip everything to have csv file we simply need to run the following through terminal:
```
find /root/MIMICIV/data/ -type f -name "*.csv.gz" -exec gzip -d {} +
```
In this case we are discarding all the .csv.gz substituting them with the uncompressed file. (Check that all the files are decompressed)

* Copy pyhealth folder from local PC to VM environment, from the local terminal (always in the MIMICIV git folder):
In this way we don't need to have clone the repo for nothing (there is no on conda)
```
scp -r ~/Documents/PHD_LLM/PyHealth/pyhealth <root@192.168.1.100>:/root/MIMICIV/
```