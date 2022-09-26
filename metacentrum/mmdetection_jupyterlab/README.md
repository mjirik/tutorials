# Learn about Metacentrum


[Metacentrum tutorial](https://wiki.metacentrum.cz/wiki/Pruvodce_pro_zacatecniky)


# Run on Metacentrum

Get the scripts to run
```bash
mkdir -p ~/projects
cd ~/projects/
git clone https://github.com/mjirik/tutorials.git
```


Run the experiment
```bash
cd ~/projects/tutorials/metacentrum/mmdetection_jupyterlab/
qsub Jupyter_job_22.01-r2.sh
```

Check email for link to you jupyterlab. In first email there is an password.
You can reset password by deleting file `$HOMEDIR/.jupyter/jupyter_notebook_config.json` and run job again with this script.


Kill the job when you are finished by `qdel` using id of your job.

```bash
qdel 12146977.meta-pbs.metacentrum.cz
```
